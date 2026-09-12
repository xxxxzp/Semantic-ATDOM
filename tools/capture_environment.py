"""Capture the runtime identity used for a Semantic A-TDOM experiment."""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

import torch


PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "pytorch3d",
    "numpy",
    "scipy",
    "scikit-image",
    "Pillow",
    "plyfile",
    "tifffile",
    "opencv-python",
    "lpips",
)

EXTENSIONS = (
    "diff_gaussian_rasterization",
    "simple_knn",
    "fused_ssim",
)


def _command(*arguments):
    result = subprocess.run(
        arguments,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "command": list(arguments),
        "exit_code": result.returncode,
        "output": result.stdout.strip(),
    }


def _package_versions():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _extension_paths():
    paths = {}
    for module_name in EXTENSIONS:
        module = importlib.import_module(module_name)
        locations = []
        if getattr(module, "__file__", None) is not None:
            locations.append(str(Path(module.__file__).resolve()))
        locations.extend(
            str(Path(item).resolve())
            for item in getattr(module, "__path__", ())
        )
        binary_name = {
            "diff_gaussian_rasterization": "diff_gaussian_rasterization._C",
            "simple_knn": "simple_knn._C",
            "fused_ssim": "fused_ssim_cuda",
        }[module_name]
        binary = importlib.import_module(binary_name)
        locations.append(str(Path(binary.__file__).resolve()))
        paths[module_name] = list(dict.fromkeys(locations))
    rasterizer = importlib.import_module("diff_gaussian_rasterization")._C
    paths["semantic_cuda_symbols"] = {
        "semantic_observation": hasattr(rasterizer, "semantic_observation"),
        "semantic_tdom": hasattr(rasterizer, "semantic_tdom"),
        "semantic_tdom_tile": hasattr(rasterizer, "semantic_tdom_tile"),
    }
    return paths


def capture():
    cuda_available = torch.cuda.is_available()
    cuda = {
        "available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if cuda_available:
        device = torch.cuda.current_device()
        cuda.update(
            {
                "visible_device_index": device,
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
        },
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda": cuda,
        },
        "packages": _package_versions(),
        "extensions": _extension_paths(),
        "git": {
            "head": _command("git", "rev-parse", "HEAD"),
            "branch": _command("git", "branch", "--show-current"),
            "status": _command("git", "status", "--short"),
        },
        "compiler": {
            "nvcc": _command("nvcc", "--version"),
            "cxx": _command("c++", "--version"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = capture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
