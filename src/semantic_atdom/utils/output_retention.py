"""Retention rules for completed progressive-training outputs."""

from pathlib import Path
import shutil


def prune_intermediate_stage_outputs(output_root, final_stage):
    """Remove numeric stage directories other than the validated final stage.

    The caller must invoke this only after every training, evaluation, and
    rendering worker has completed successfully.  Non-numeric output products
    (maps, renders, semantic state, logs, and caches) are never considered.
    """

    root = Path(output_root).expanduser().resolve()
    final_name = str(int(final_stage))
    final_path = root / final_name
    if not final_path.is_dir() or final_path.is_symlink():
        raise RuntimeError(
            "cannot prune stage outputs without a real final-stage directory: "
            + str(final_path)
        )

    candidates = []
    for child in root.iterdir():
        if not child.name.isdecimal() or child.name == final_name:
            continue
        if child.is_symlink():
            raise RuntimeError(
                "refusing to prune a numeric stage symlink: " + str(child)
            )
        if child.is_dir():
            candidates.append(child)

    for child in candidates:
        shutil.rmtree(child)
    return tuple(sorted(int(child.name) for child in candidates))
