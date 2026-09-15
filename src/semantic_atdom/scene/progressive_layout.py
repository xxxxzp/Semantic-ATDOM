"""Resolve supported progressive A-TDOM dataset layouts without mutating inputs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})


@dataclass(frozen=True)
class ProgressiveDatasetLayout:
    scene_root: Path
    progressive_root: Path
    image_root: Optional[Path]


def select_progressive_stages(
    progressive_root: Path, start_stage: int, end_stage: int
) -> Tuple[int, ...]:
    """Select numeric stage directories with one consistent open-end rule."""
    progressive_root = Path(progressive_root)
    available = sorted(
        int(path.name)
        for path in progressive_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    effective_end = max(available) if int(end_stage) < int(start_stage) else int(end_stage)
    return tuple(
        stage
        for stage in available
        if int(start_stage) <= stage <= effective_end
    )


def progressive_stage_number(stage_path: Path) -> int:
    """Resolve the authoritative stage number from a stage output/input path."""
    name = Path(stage_path).name
    if not name.isdigit():
        raise ValueError("progressive stage path must end in a number: {}".format(stage_path))
    return int(name)


def _contains_images(path: Path) -> bool:
    try:
        return any(
            item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES
            for item in path.iterdir()
        )
    except OSError:
        return False


def _shared_image_candidates(scene_root: Path) -> Tuple[Path, ...]:
    try:
        children = tuple(scene_root.iterdir())
    except OSError:
        return ()
    return tuple(
        sorted(
            (
                item.resolve()
                for item in children
                if item.is_dir()
                and item.name.endswith("_CAM")
                and _contains_images(item)
            ),
            key=lambda item: item.name,
        )
    )


def resolve_progressive_dataset_layout(
    source_root: Path, image_folder_path: Optional[Path] = None
) -> ProgressiveDatasetLayout:
    """Resolve legacy stage roots and the fixed ``GaussianSplatting`` layout.

    Fixed-layout scenes keep all RGB frames in one sibling ``*_CAM`` directory.
    An explicit image folder always wins; implicit discovery is accepted only
    when exactly one candidate exists.
    """
    source_root = Path(source_root).expanduser().resolve()
    progressive_child = source_root / "GaussianSplatting"
    if progressive_child.is_dir():
        scene_root = source_root
        progressive_root = progressive_child.resolve()
    else:
        scene_root = source_root.parent
        progressive_root = source_root

    explicit = None
    if image_folder_path is not None and str(image_folder_path).strip():
        explicit = Path(image_folder_path).expanduser().resolve()
        if not explicit.is_dir():
            raise ValueError("image folder does not exist: {}".format(explicit))

    image_root = explicit
    if image_root is None and progressive_child.is_dir():
        candidates = _shared_image_candidates(scene_root)
        if len(candidates) == 1:
            image_root = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                "multiple shared *_CAM image folders found; set "
                "--Image_Folder_Path explicitly: {}".format(
                    [str(item) for item in candidates]
                )
            )

    return ProgressiveDatasetLayout(
        scene_root=scene_root,
        progressive_root=progressive_root,
        image_root=image_root,
    )
