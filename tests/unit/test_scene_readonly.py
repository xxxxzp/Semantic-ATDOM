import hashlib
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from PIL import Image
from semantic_atdom.scene.dataset_readers import readColmapSceneInfo


def source_fixture(root):
    sparse = root / 'sparse' / '0'
    sparse.mkdir(parents=True)
    (root / 'images').mkdir()
    with (sparse / 'cameras.bin').open('wb') as stream:
        stream.write(struct.pack('<QiiQQdddd', 1, 1, 1, 4, 4, 3., 3., 2., 2.))
    with (sparse / 'images.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', 2))
        for index in (1, 2):
            stream.write(struct.pack('<idddddddi', index, 1., 0., 0., 0., float(index), 0., 0., 1))
            stream.write(f'{index:06d}.png'.encode() + b'\0')
            stream.write(struct.pack('<Q', 0))
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(root / 'images' / f'{index:06d}.png')
    with (sparse / 'points3D.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', 1))
        stream.write(struct.pack('<QdddBBBdQ', 1, 0., 0., 3., 100, 120, 140, 0., 0))


def contents(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class SceneReadOnlyTests(unittest.TestCase):
    def test_default_colmap_reader_keeps_source_tree_unchanged(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'raw'
            source_fixture(source)
            before = contents(source)
            scene = readColmapSceneInfo(str(source), None, False, No_Key_Region=True)
            self.assertIsNotNone(scene.point_cloud)
            self.assertEqual(contents(source), before)
            self.assertIsNone(scene.ply_path)

    def test_explicit_output_cache_does_not_write_source(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'raw'
            source_fixture(source)
            before = contents(source)
            cache = Path(directory) / 'output' / 'points.ply'
            scene = readColmapSceneInfo(str(source), None, False, No_Key_Region=True, Ply_Cache_Path=cache)
            self.assertEqual(contents(source), before)
            self.assertTrue(cache.is_file())
            self.assertIsNotNone(scene.point_cloud)
