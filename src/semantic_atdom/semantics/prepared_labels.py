"""Bounded CPU reuse of geometry-independent semantic raster preparation."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np

from semantic_atdom.semantics.label_grid import camera_mask, rectified_source_labels
from semantic_atdom.utils.pixel_grid import PIXEL_GRID_VERSION, resize_nearest


def _digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


class PreparedLabelCache:
    """Cache only derived CPU arrays; callers still load/validate each input.

    Content, calibration, mask, target grid and reliability settings identify
    entries. Extrinsics and Gaussian geometry intentionally do not: every call
    must rasterize those again. Disk files live only in the run output directory.
    The owning semantic worker serializes access.
    """

    def __init__(self, directory=None, max_bytes=128 * 1024**2,
                 max_disk_bytes=512 * 1024**2):
        self.directory = None if directory is None else Path(directory)
        self.max_bytes = max(0, int(max_bytes))
        self._entries = OrderedDict()
        self.nbytes = 0
        self.max_disk_bytes = max(0, int(max_disk_bytes))
        self._disk = OrderedDict()
        if self.directory is not None and self.directory.is_dir():
            for path in sorted(self.directory.glob('*.npz'), key=lambda p: p.stat().st_mtime_ns):
                self._disk[path.name] = path.stat().st_size
        self.stats = dict(memory_hits=0, disk_hits=0, misses=0, corrupt_entries=0,
                          write_errors=0, retained_bytes=0)
        self._trim_disk()

    def prepare(self, labels, camera, ignore_id, bandwidth, floor, boundary_fn):
        calibration = getattr(camera, 'calibration', None)
        mask = camera_mask(camera)
        resolution = (int(camera.image_width), int(camera.image_height))
        identity = dict(version=1, pixel_grid=PIXEL_GRID_VERSION,
            labels=_digest(labels), mask=None if mask is None else _digest(mask),
            calibration=None if calibration is None else calibration.to_dict(),
            resolution=resolution, ignore_id=int(ignore_id),
            bandwidth=float(bandwidth), floor=float(floor))
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if key in self._entries:
            self.stats['memory_hits'] += 1
            self._entries.move_to_end(key)
            return self._entries[key]
        path = None if self.directory is None else self.directory / (key + '.npz')
        value = None
        if path is not None and path.is_file():
            try:
                with np.load(path, allow_pickle=False) as archive:
                    ids, weights = archive['ids'], archive['weights']
                    metadata = json.loads(str(archive['metadata']))
                if (metadata['key'] != key or ids.shape != resolution[::-1]
                    or weights.shape != ids.shape or weights.dtype != np.float32
                    or not np.issubdtype(ids.dtype, np.integer)
                    or _digest(ids) != metadata['ids_sha256']
                    or _digest(weights) != metadata['weights_sha256']):
                    raise ValueError('prepared raster identity mismatch')
                value = (ids, weights)
                self.stats['disk_hits'] += 1
                if path.name in self._disk:
                    self._disk.move_to_end(path.name)
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                self.stats['corrupt_entries'] += 1
        if value is None:
            self.stats['misses'] += 1
            source = rectified_source_labels(labels, camera, ignore_id)
            weights = boundary_fn(source, ignore_id=ignore_id, bandwidth=bandwidth, floor=floor)
            value = (resize_nearest(source, resolution), resize_nearest(weights, resolution))
            if path is not None:
                self._write(path, key, value)
        for array in value:
            array.flags.writeable = False
        size = sum(array.nbytes for array in value)
        if size <= self.max_bytes:
            while self._entries and self.nbytes + size > self.max_bytes:
                _, previous = self._entries.popitem(last=False)
                self.nbytes -= sum(array.nbytes for array in previous)
            self._entries[key] = value
            self.nbytes += size
        self.stats['retained_bytes'] = self.nbytes
        return value

    def _write(self, path, key, value):
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            ids, weights = value
            metadata = json.dumps(dict(key=key, ids_sha256=_digest(ids), weights_sha256=_digest(weights)))
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, ids=ids, weights=weights, metadata=metadata)
            os.replace(temporary, path)
            self._disk[path.name] = path.stat().st_size
            self._disk.move_to_end(path.name)
            self._trim_disk()
        except OSError:
            self.stats['write_errors'] += 1
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    self.stats['write_errors'] += 1

    def _trim_disk(self):
        try:
            while self._disk and sum(self._disk.values()) > self.max_disk_bytes:
                name, _ = self._disk.popitem(last=False)
                (self.directory / name).unlink(missing_ok=True)
        except OSError:
            self.stats['write_errors'] += 1
        self.stats['disk_bytes'] = sum(self._disk.values())
