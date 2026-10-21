import os
import hashlib
import csv
from queue import Empty
from semantic_atdom.utils.progress_capture import ProgressCapture
from semantic_atdom.semantics.orchestrator import FrozenCameraSnapshot
from semantic_atdom.utils.worker_supervision import WorkerGroup
import random
from pathlib import Path
import torch
from random import randint
from semantic_atdom.utils.loss_utils import l1_loss
from semantic_atdom.gaussian_renderer import render
import sys
from semantic_atdom.scene import Scene, GaussianModel
from fused_ssim import fused_ssim
from semantic_atdom.utils.general_utils import safe_state
from tqdm import tqdm
from semantic_atdom.utils.image_utils import psnr, tdom_to_uint8
from semantic_atdom.utils.metrics import lpips_distance, ssim_chw
from semantic_atdom.utils.training_schedule import (
    progressive_stage_iterations,
    total_training_iterations,
)
from semantic_atdom.utils.output_retention import (
    prune_intermediate_stage_outputs,
)
from semantic_atdom.utils.run_output import (
    RunLayout,
    allocate_run_root,
    archive_semantic_state,
    finalize_stage_model,
    relative_artifact_records,
    sha256_file,
    validate_camera_render_outputs,
    validate_final_semantic_outputs,
    write_json_atomic,
)
from argparse import ArgumentParser, Namespace
from semantic_atdom.arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial import cKDTree
import math
import time
from semantic_atdom.scene.cameras import Camera, DummyCamera, DummyPipeline
from semantic_atdom.utils.camera_utils import camera_to_JSON, resize_camera_mask, loadCam
from semantic_atdom.utils.camera_calibration import projection_matches_retained
from semantic_atdom.utils.pixel_grid import PIXEL_GRID_VERSION
import json
from semantic_atdom.scene.dataset_readers import (
    load_rectified_image,
    mask_preparation_statistics,
    sceneLoadTypeCallbacks,
)
from semantic_atdom.scene.progressive_layout import (
    progressive_stage_number,
    resolve_progressive_dataset_layout,
    select_progressive_stages,
)
from semantic_atdom.utils.general_utils import PILtoTorch
import numpy as np
import shutil
import glob
import cv2
import gc
import statistics
import copy
import warnings
import subprocess
import traceback
from semantic_atdom.utils.graphics_utils import getWorld2View2, getProjectionMatrix
import torchvision
import torch.multiprocessing as mp
import lpips
from semantic_atdom.semantics.orchestrator import (
    AsyncProgressiveSemanticHook,
    MapPublicationRequest,
    OnlineSemanticConfig,
)
from semantic_atdom.semantics.planar_fallback import PlanarFallbackConfig
from semantic_atdom.semantics.publication import (
    MapGridSpec,
    MapPublicationPolicy,
)
from semantic_atdom.semantics.schema import SemanticClassSpec
from semantic_atdom.semantics.preflight import run_dataset_preflight
from semantic_atdom.semantics.tdom import (
    MAX_SEMANTIC_TDOM_CLASSES,
    SemanticTdomConfig,
    load_semantic_tdom_palette,
    rasterize_rgb_tdom_striped,
    rasterize_semantic_tdom,
)
from semantic_atdom.training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    camera_from_state,
    camera_to_state,
    capture_rng_state,
    checkpoint_stage_from_path,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

WARNED = False

# Maximum encoded width.  Portrait map footprints use a narrower, content-fit
# canvas so the paired maps are not separated by empty half-panel padding.
TRAINING_PROGRESS_WIDTH = 3840
TRAINING_PROGRESS_MIN_WIDTH = 1280
TRAINING_PROGRESS_HEIGHT = 1080
TRAINING_PROGRESS_MAP_DOWNSAMPLE = 8
TRAINING_PROGRESS_UPDATE_INTERVAL = 10


def camera_image_mapping_changed(camera_info, previous):
    """Return whether a retained RGB tensor needs the new rectification map."""

    return not projection_matches_retained(
        camera_info.FovX, camera_info.FovY, camera_info.calibration, previous)



def trainable_cameras(cameras):
    return [camera for camera in cameras if camera.Tri_Mask.shape[0] != 2]


class StageTrainingInputCache:
    """Keep repeatedly sampled stage inputs on the training device."""

    def __init__(self, cameras, device):
        self.device = device
        self._images = {
            id(camera): camera.original_image.to(device) for camera in cameras
        }
        self._masks = {
            id(camera): camera.Tri_Mask.to(device) for camera in cameras
        }

    def get(self, camera):
        cached = self._images.get(id(camera))
        if cached is not None:
            return cached
        return camera.original_image.to(self.device)

    def get_mask(self, camera):
        cached = self._masks.get(id(camera))
        if cached is not None:
            return cached
        return camera.Tri_Mask.to(self.device)

    def clear(self):
        self._images.clear()
        self._masks.clear()


class TrainingProgressReporter:
    """Rate-limit host synchronization while keeping tqdm progress exact."""

    def __init__(self, progress_bar, interval=10, initial_ema_loss=0.0):
        if interval <= 0:
            raise ValueError("progress interval must be positive")
        self.progress_bar = progress_bar
        self.interval = int(interval)
        self.ema_loss = float(initial_ema_loss)
        self._reported_steps = 0

    def maybe_update(self, loss, step, total_steps):
        if step % self.interval != 0 and step != total_steps:
            return False

        self.ema_loss = 0.4 * loss.item() + 0.6 * self.ema_loss
        self.progress_bar.update(step - self._reported_steps)
        self.progress_bar.set_postfix({"Loss": f"{self.ema_loss:.7f}"})
        self._reported_steps = step
        if step == total_steps:
            self.progress_bar.close()
        return True


def release_camera_sampling_buffers(cameras):
    """Release LoG sampling data after sparse-point generation consumes it."""
    for camera in cameras:
        camera.GradLog = None
        camera.MaxGradLog = -1
        camera.LogMask = None


def should_retain_visibility(training_type, no_sampling, reset_pending=False):
    """Return whether a later opacity reset can consume per-view visibility."""
    return (
        training_type == "Initialization"
        or bool(no_sampling)
        or bool(reset_pending)
    )


def summarize_progressive_stage_timings(stage_records):
    """Summarize progressive latency using actual newly introduced images."""
    image_count = sum(int(item["new_image_count"]) for item in stage_records)
    if image_count <= 0:
        return {
            "progressive_stage_count": len(stage_records),
            "progressive_image_count": 0,
            "geometry_seconds_per_image": None,
            "online_seconds_per_image": None,
        }

    geometry_seconds = sum(
        float(item["geometry_seconds"]) for item in stage_records
    )
    online_seconds = sum(
        float(item["online_seconds"]) for item in stage_records
    )
    return {
        "progressive_stage_count": len(stage_records),
        "progressive_image_count": image_count,
        "geometry_seconds_per_image": geometry_seconds / image_count,
        "online_seconds_per_image": online_seconds / image_count,
    }


def select_training_progress_stages(start_stage, end_stage, target_frames):
    """Choose online stage boundaries while reserving one final-refinement frame."""
    start_stage = int(start_stage)
    end_stage = int(end_stage)
    target_frames = int(target_frames)
    if end_stage < start_stage:
        raise ValueError("end stage must not precede start stage")
    if target_frames < 3:
        raise ValueError("target frame count must be at least 3")
    available = end_stage - start_stage + 1
    progressive_budget = min(available, target_frames - 1)
    if progressive_budget == available:
        return tuple(range(start_stage, end_stage + 1))
    positions = np.linspace(
        start_stage, end_stage, num=progressive_budget, endpoint=True
    )
    selected = {int(round(value)) for value in positions}
    selected.update((start_stage, end_stage))
    return tuple(sorted(selected))

def SemanticWarning(message):
    """Warnings-as-errors must never couple semantics back into geometry."""
    try:
        warnings.warn(message, RuntimeWarning)
    except Exception:
        pass


def resolve_semantic_render_palette(semantic_data_root, explicit_path=None):
    semantic_root = Path(semantic_data_root) / "semantic"
    class_spec = SemanticClassSpec.load(semantic_root / "class_spec.json")
    colors = tuple(item.color for item in class_spec.classes)
    palette_path = explicit_path
    if not palette_path:
        candidate = semantic_root / "tdom_palette.json"
        if candidate.is_file():
            palette_path = str(candidate)
    palette_source = None
    if palette_path:
        try:
            colors = load_semantic_tdom_palette(palette_path, class_spec)
            palette_source = str(palette_path)
        except Exception as error:
            SemanticWarning(
                "Semantic TDOM palette ignored; using class_spec colors: {}".format(
                    error
                )
            )
    return class_spec, colors, palette_source

def Get_lpips(rgbs: torch.Tensor, target_rgbs: torch.Tensor, lpips_vgg):
    return {'vgg': lpips_distance(target_rgbs, rgbs, lpips_vgg)}

def prepare_output_and_logger(args, Create_for_new=False, Silence=False):
    # 创建对应的输出文件夹
    if not Create_for_new:
        m_path = args.model_path
    else:
        m_path = args.model_path_second

    if not Silence:
        print("Output folder: {}".format(m_path))
    os.makedirs(m_path, exist_ok=True)

    # 在文件夹中创建并打开一个二进制文件cfg_args，并在里面输出参数配置其的所有内容
    with open(os.path.join(m_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建一个Tensorboard writer对象
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(m_path)
    else:
        if not Silence:
            print("Tensorboard not available: not logging progress")
    return tb_writer

def GetArgs():
    '''以下是在构建参数配置器'''
    # 构建一个空的原始参数配置器
    parser = ArgumentParser(description="Training script parameters")

    # 构建一个用于存储与模型有关参数的参数配置器
    lp = ModelParams(parser)

    # 构建一个用于存储与模型优化有关参数的参数配置器
    op = OptimizationParams(parser)

    # 构建一个用于存储与流程处理有关参数的参数配置器
    pp = PipelineParams(parser)

    # 为原始参数配置器加入一些参数
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)  # 是否开启某些异常检测，默认为False
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")  # quiet默认为False，当设置为True时，则在训练过程中不会输出任何内容到日志文件
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--render_all_cameras", action="store_true")
    parser.add_argument("--render_training_progress", action="store_true")
    parser.add_argument(
        "--training_progress_target_frames", type=int, default=120
    )
    parser.add_argument("--worker_shutdown_timeout", type=float, default=300.0,
                        help="Maximum seconds to drain each worker during shutdown.")
    parser.add_argument("--training_progress_fps", type=float, default=4.0)
    parser.add_argument(
        "--recovery_checkpoint_every_stages", type=int, default=0
    )
    parser.add_argument("--save_final_training_state", action="store_true")
    parser.add_argument(
        "--skip_final_evaluation", action="store_true",
        help="Skip final image quality metrics; still publish models, maps and video.",
    )

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.skip_final_evaluation and args.render_all_cameras:
        raise ValueError(
            "--skip_final_evaluation cannot be combined with --render_all_cameras "
            "because camera exports currently run inside final evaluation"
        )
    '''以上是在构建参数配置器'''

    if not math.isfinite(args.worker_shutdown_timeout) or args.worker_shutdown_timeout <= 0:
        raise ValueError("--worker_shutdown_timeout must be positive and finite")
    if args.training_progress_target_frames < 3:
        raise ValueError(
            "--training_progress_target_frames must be at least 3"
        )
    if args.training_progress_fps <= 0:
        raise ValueError("--training_progress_fps must be positive")
    if args.recovery_checkpoint_every_stages < 0:
        raise ValueError(
            "--recovery_checkpoint_every_stages must be non-negative"
        )
    if args.render_training_progress and not args.enable_semantics:
        raise ValueError(
            "--render_training_progress requires --enable_semantics"
        )
    if args.render_training_progress and not args.render_TDOM:
        raise ValueError(
            "--render_training_progress requires --render_TDOM"
        )
    if args.OutputModelInterval:
        raise ValueError(
            "--OutputModelInterval has been replaced by "
            "--recovery_checkpoint_every_stages"
        )
    if args.render_RGB:
        raise ValueError(
            "--render_RGB is unsupported; use --render_training_progress "
            "for the synchronized RGB/Semantic TDOM timeline"
        )
    if args.run_name:
        allocated = os.environ.get("SEMANTIC_ATDOM_RUN_ROOT")
        if allocated is None:
            allocated = str(
                allocate_run_root(
                    args.Model_Path_Dir,
                    args.run_name,
                    repository_root=Path(__file__).resolve().parent.parent,
                )
            )
            os.environ["SEMANTIC_ATDOM_RUN_ROOT"] = allocated
        args.Model_Path_Dir = allocated
    layout = RunLayout.from_root(args.Model_Path_Dir)
    args.run_root = str(layout.root)
    args.work_dir = str(layout.work)
    args.stage_output_dir = str(layout.stages)
    args.semantic_work_dir = str(layout.semantic_work)
    args.map_output_dir = str(layout.maps)
    args.metrics_dir = str(layout.metrics)
    args.logs_dir = str(layout.logs)
    args.provenance_dir = str(layout.provenance)
    args.renders_dir = str(layout.renders)
    args.videos_dir = str(layout.videos)

    # 输出存储训练后模型的存储位置
    print("Optimizing " + args.Source_Path_Dir)
    layout = resolve_progressive_dataset_layout(
        args.Source_Path_Dir, args.Image_Folder_Path or None
    )
    args.Source_Path_Dir = str(layout.progressive_root)
    if not args.Image_Folder_Path and layout.image_root is not None:
        args.Image_Folder_Path = str(layout.image_root)
        print("Discovered shared RGB folder: " + args.Image_Folder_Path)

    if args.start_checkpoint:
        args.start_checkpoint = str(Path(args.start_checkpoint).expanduser().resolve())
        resume_stage = checkpoint_stage_from_path(Path(args.start_checkpoint))
        if args.ContinueFromImageNo not in (-1, resume_stage):
            raise ValueError(
                "--ContinueFromImageNo disagrees with --start_checkpoint"
            )
        args.ContinueFromImageNo = resume_stage
    elif args.ContinueFromImageNo != -1:
        args.start_checkpoint = os.path.join(
            args.work_dir,
            "recovery",
            "latest.pt",
        )
    if args.start_checkpoint and not os.path.isfile(args.start_checkpoint):
        raise FileNotFoundError(
            "resume checkpoint does not exist: {}".format(args.start_checkpoint)
        )

    return args, lp, op, pp


def semantic_online_required(args):
    """Whether a requested product consumes progressive semantic state."""
    return bool(
        args.enable_semantics
        and (
            not args.semantic_tdom_final_only
            or args.semantic_checkpoint_interval > 0
            or args.render_training_progress
        )
    )

def TrainingPreparation(args):
    TimeCost = {}

    if args.enable_semantics and not str(args.semantic_data_root).strip():
        raise ValueError(
            "--semantic_data_root is required when --enable_semantics is set"
        )
    if (
        args.enable_semantics
        and args.resolution not in (1, 2, 4, 8)
        and args.resolution < 9
    ):
        raise ValueError(
            "semantic projection requires --resolution 1, 2, 4 or 8, or an "
            "explicit target width greater than 8; automatic resizing is unsupported"
        )
    if args.checkpoint_iterations:
        raise ValueError(
            "iteration checkpoints are not resumable at an arbitrary optimizer "
            "step; use --recovery_checkpoint_every_stages for atomic "
            "stage-boundary recovery"
        )

    args.pixel_grid_version = PIXEL_GRID_VERSION

    # 输出训练日志
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    if args.ContinueFromImageNo == -1:
        Diary = open(os.path.join(args.logs_dir, "training.log"), "w")
        EvaluateDiary = open(os.path.join(args.metrics_dir, "evaluations.jsonl"), "w")
    else:
        Diary = open(os.path.join(args.logs_dir, "training.log"), "a")
        EvaluateDiary = open(os.path.join(args.metrics_dir, "evaluations.jsonl"), "a")
    GaussianDiary = None

    # 这个字典用于存储每一张影像已经被训练了多少次
    ImagesAlreadyBeTrainedIterations = {}

    # 一些输出上的设置
    safe_state(args.quiet, args.gpu)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 读取所有的source_path
    IM = list(
        select_progressive_stages(
            args.Source_Path_Dir,
            args.StartFromImageNo,
            args.EndAtImageNo,
        )
    )
    if not IM:
        raise ValueError("no progressive stages match the requested range")
    if args.ContinueFromImageNo != -1 and args.ContinueFromImageNo not in IM:
        raise ValueError("resume stage is outside the selected progressive range")
    dataset_id = hashlib.sha256(
        os.path.abspath(args.Source_Path_Dir).encode("utf-8")
    ).hexdigest()[:16]
    args.dataset_cache_root = os.path.join(
        args.work_dir, "cache", dataset_id
    )
    if args.enable_semantics:
        preflight = run_dataset_preflight(
            dataset_root=args.Source_Path_Dir,
            semantic_data_root=args.semantic_data_root,
            output_root=args.work_dir,
            start_stage=args.StartFromImageNo,
            end_stage=args.EndAtImageNo,
            image_folder_path=(args.Image_Folder_Path or None),
            identity_policy=args.progressive_identity_policy,
            # Recovery keeps the original stage range and input contract.
            # Changing this policy bit changes the hashed preflight report
            # and rejects a checkpoint even when every input is unchanged.
            require_complete_initial_snapshot=True,
            max_semantic_tdom_classes=(
                MAX_SEMANTIC_TDOM_CLASSES if args.render_TDOM else None
            ),
        )
        provenance_preflight = Path(args.provenance_dir) / "dataset_preflight.json"
        provenance_preflight.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preflight.report_path, provenance_preflight)
        args.dataset_preflight_path = str(provenance_preflight)
        args.dataset_preflight_sha256 = preflight.report_sha256
        args.dataset_cache_root = str(preflight.report_path.parent)
        args.semantic_authoritative_exact_complete = (
            preflight.authoritative_exact_complete
        )
        print(
            "Dataset preflight: stages={}..{}, labels_complete={}, "
            "source_binding_complete={}, sha256={}".format(
                preflight.selected_stages[0],
                preflight.selected_stages[-1],
                preflight.labels_complete,
                preflight.source_binding_complete,
                preflight.report_sha256,
            )
        )
        with open(preflight.report_path, "r", encoding="utf-8") as stream:
            preflight_payload = json.load(stream)
        args.progressive_stage_new_counts = tuple(
            len(record["new_image_ids"])
            for record in preflight_payload["stage_records"][1:]
        )
    print(IM)
    source_path_list = [os.path.join(args.Source_Path_Dir, str(im)) for im in IM]
    model_path_list = [os.path.join(args.stage_output_dir, str(im)) for im in IM]
    Diary.write(f"source_path_Dir: {args.Source_Path_Dir}\nModel_Path_Dir: {args.Model_Path_Dir}\n")
    Diary.write(f"Progress: {IM}\n")
    Diary.write("\n")

    Diary.write(f"Mean: {args.Mean}, "
                f"MergeScene_Densification_Interval: {args.MergeScene_Densification_Interval}, "
                f"OpacityThreshold: {args.opacity_threshold}, "
                f"InitialTrainingTimesSetZero: {args.InitialTrainingTimesSetZero}, "
                f"UseDifferentImageLr: {args.DifferentImagesLr}, "
                f"UseDepthLoss: {args.UseDepthLoss}, "
                f"UseScaleLoss: {args.UseScaleLoss}, "
                f"UseNormalLoss: {args.GetNormal}\n")
    print(f"Mean: {args.Mean}, "
          f"MergeScene_Densification_Interval: {args.MergeScene_Densification_Interval}, "
          f"OpacityThreshold: {args.opacity_threshold}, "
          f"InitialTrainingTimesSetZero: {args.InitialTrainingTimesSetZero}, "
          f"UseDifferentImageLr: {args.DifferentImagesLr}, "
          f"UseDepthLoss: {args.UseDepthLoss}, "
          f"UseScaleLoss: {args.UseScaleLoss}, "
          f"UseNormalLoss: {args.GetNormal}")

    # 如果使用原本的学习率更新方法，则重新设置position_lr_max_step
    if not args.DifferentImagesLr:
        new_counts = getattr(
            args,
            "progressive_stage_new_counts",
            tuple(max(0, second - first) for first, second in zip(IM, IM[1:])),
        )
        WholeIterations = total_training_iterations(
            initial_iterations=args.IterationFirstScene,
            final_iterations=args.FinalOptimizationIterations,
            stage_new_image_counts=new_counts,
            local_iterations_per_image=args.IterationPerMergeScene,
            global_interval=args.GlobalOptimizationInterval,
            global_iterations=args.GlobalOptimizationIteration,
        )
        args.position_lr_max_steps = WholeIterations

    return TimeCost, Diary, EvaluateDiary, GaussianDiary, ImagesAlreadyBeTrainedIterations, source_path_list, model_path_list, IM


def copy_dataset_context(source_args, worker_args):
    """Copy preflight/cache identity to an independently parsed worker."""
    if not hasattr(source_args, "dataset_cache_root"):
        raise AttributeError(
            "source arguments have not completed TrainingPreparation"
        )
    context_fields = (
        "dataset_cache_root",
        "dataset_preflight_path",
        "dataset_preflight_sha256",
        "semantic_authoritative_exact_complete",
        "progressive_stage_new_counts",
        "position_lr_max_steps",
    )
    for field in context_fields:
        if hasattr(source_args, field):
            setattr(worker_args, field, getattr(source_args, field))

def str2dict(str):
    replacestr = "{}\':,\n"
    for rstr in replacestr:
        str = str.replace(rstr, '')

    list = str.split(' ')
    dict = {}
    for i in range(int(len(list) / 2)):
        if '.' not in list[2 * i + 1]:
            dict[list[2 * i]] = int(list[2 * i + 1])
        else:
            dict[list[2 * i]] = float(list[2 * i + 1])
    return dict

def str2list(str):
    replacestr = "[]\',\n"
    for rstr in replacestr:
        str = str.replace(rstr, '')

    list = str.split(' ') if str != "" else []
    return list

def Get_LogImage(image, image_name="", OutputDirPath="", sigma=1.0, MaxGradLog=-1, save=False):
    # ---- 1️⃣ 确保为numpy数组 ----
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    # ---- 2️⃣ 自动判断通道顺序 ----
    if image.ndim == 3:
        if image.shape[0] == 3:  # (3, H, W)
            R, G, B = image[0], image[1], image[2]
        elif image.shape[2] == 3:  # (H, W, 3)
            R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        else:
            raise ValueError(f"Unexpected image shape {image.shape}")
    else:
        raise ValueError(f"Invalid image dimensions: {image.shape}")

    # ---- 3️⃣ 计算灰度图 ----
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray = gray.astype(np.float32)

    # ---- 4️⃣ 转换并归一化到0-255 ----
    image = np.clip(gray * 255, 0, 255).astype(np.uint8)

    # ---- 5️⃣ 高斯模糊 + LoG ----
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log_abs = np.abs(log)

    # ---- 6️⃣ 归一化 ----
    if MaxGradLog <= 0 or np.max(log_abs) >= MaxGradLog:
        log_norm = log_abs / np.max(log_abs)
        MG = np.max(log_abs)
    else:
        log_norm = log_abs / MaxGradLog
        MG = MaxGradLog

    psi = np.minimum(log_norm, 1.0)
    psi_inverted = psi

    # ---- 7️⃣ 保存结果 ----
    if save and OutputDirPath and image_name:
        os.makedirs(OutputDirPath, exist_ok=True)
        save_path = os.path.join(OutputDirPath, f"{image_name}.jpg")
        cv2.imwrite(save_path, (psi_inverted * 255).astype(np.uint8))

    return psi, MG

def Get_LogImage_origin(image, image_name="", OutputDirPath="", sigma=1.0, MaxGradLog=-1, save=False):
    R, G, B = image[0], image[1], image[2]
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    image = gray.squeeze(0).cpu().numpy()
    image = (image * 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log_abs = np.abs(log)
    # log_norm = log_abs
    if np.max(log_abs) >= MaxGradLog:
        log_norm = log_abs / np.max(log_abs)
        MG = np.max(log_abs)
    else:
        log_norm = log_abs / MaxGradLog
        MG = MaxGradLog

    psi = np.minimum(log_norm, 1.0)
    psi_inverted = psi

    if save and OutputDirPath != "" and image_name != "":
        os.makedirs(OutputDirPath, exist_ok=True)
        cv2.imwrite(os.path.join(OutputDirPath, f"{image_name}.jpg"), (psi_inverted * 255).astype(np.uint8))

    return psi, MG

def LogImage_minus(Log1, Log2, save_Dir, image_name, save=False):
    psi_B = np.maximum(Log1 - Log2, 0)

    if save:
        os.makedirs(save_Dir, exist_ok=True)

        save_path = os.path.join(save_Dir, image_name + ".jpg")
        cv2.imwrite(save_path, (psi_B * 255).astype(np.uint8))

    return psi_B

def remove_outliers_quantile(x, y, q=0.995):
    """
    x, y: (N,) torch.tensor on CUDA
    q:    保留的分位数，0.99 ~ 0.999 都常用
    """
    x_min, x_max = torch.quantile(x, torch.tensor([1-q, q], device=x.device))
    y_min, y_max = torch.quantile(y, torch.tensor([1-q, q], device=y.device))

    mask = (x >= x_min) & (x <= x_max) & \
           (y >= y_min) & (y <= y_max)
    return mask

def camera_orthogonal_bounding_rectangle(points, R, T):
    """
    points: (N, 3) torch tensor
    R: (3, 3) torch tensor (world from camera)
    T: (3,) torch tensor
    """

    with torch.no_grad():

        device = points.device
        dtype = points.dtype

        # ---- 1. camera forward direction ----

        R = R.T.to(device=device, dtype=dtype)
        T = T.to(device=device, dtype=dtype)

        d = R[:, 2]
        d = d / torch.norm(d)
        print(d)
        print(torch.dot(d, points.mean(dim=0) - T))

        # ---- 2. plane basis (u, v) ----
        tmp = torch.tensor([1.0, 0.0, 0.0], device=points.device, dtype=dtype)
        if torch.abs(torch.dot(tmp, d)) > 0.9:
            tmp = torch.tensor([0.0, 1.0, 0.0], device=points.device, dtype=dtype)

        u = torch.cross(d, tmp)
        u = u / torch.norm(u)
        v = torch.cross(d, u)

        # ---- 3. project points to plane ----
        # P = points - T[None, :]
        P = points
        x = P @ u
        y = P @ v

        mask = remove_outliers_quantile(x, y, q=0.9995)
        x = x[mask]
        y = y[mask]

        # x_np = x.detach().cpu().numpy()
        # y_np = y.detach().cpu().numpy()
        #
        # plt.figure(figsize=(6, 6))
        # plt.scatter(x_np, y_np, s=1)
        # plt.gca().set_aspect('equal')
        # plt.xlabel("u axis")
        # plt.ylabel("v axis")
        # plt.title("Orthographic projection onto plane")
        # plt.savefig("scatter.jpg", dpi=300)

        # ---- 4. 2D bounding box ----
        xmin, xmax = x.min().item(), x.max().item()
        ymin, ymax = y.min().item(), y.max().item()

        width = xmax - xmin
        height = ymax - ymin

        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5

        # ---- 5. rectangle center on ray ----
        center_plane = T + cx * u + cy * v
        t = torch.dot(center_plane - T, d)
        center = T + t * d

        # ---- 6. rectangle corners ----
        hw, hh = width * 0.5, height * 0.5
        corners = torch.stack([
            center + hw*u + hh*v,
            center + hw*u - hh*v,
            center - hw*u - hh*v,
            center - hw*u + hh*v
        ])

    return {
        "center": center,
        "normal": d,
        "u": u,
        "v": v,
        "width": height,
        "height": width,
        "corners": corners,
        "xy_max_min": [ymin, ymax, xmin, xmax],
        "Gaussians_center": [(ymin + ymax) / 2, -(xmin + xmax) / 2]
    }

def expand_image_gray(img_path, save_path, width0, height0, gray_value=128):
    """
    将图像向左、向上扩张，并用灰色填充
    """
    img = Image.open(img_path)
    width, height = img.size

    if width0 < width or height0 < height:
        raise ValueError("width0 和 height0 必须不小于原图尺寸")

    # 创建灰色背景
    background = Image.new(
        mode=img.mode,
        size=(width0, height0),
        color=(gray_value, gray_value, gray_value)
    )

    # 原图放在右下角
    offset_x = width0 - width
    offset_y = height0 - height

    background.paste(img, (offset_x, offset_y))
    background.save(save_path)

    # print(f"Saved expanded image to {save_path}")

# 前端，第一个线程，主要用于读取数据以及一些数据预处理
class DataPreProccesser(mp.Process):
    # 构造函数
    def __init__(self, args=None, lp=None, op=None, pp=None):
        super().__init__()

        self.Silence = True

        # 用于两个进程之间的信息传递
        self.DataPreProccesser_queue = None
        self.GaussianTrainer_queue = None
        self.GaussianTrainer_process = None

        # 获取参数配置器
        if args is None:
            self.args, self.lp, self.op, self.pp = GetArgs()
        else:
            self.args = copy.deepcopy(args)
            self.lp, self.op, self.pp = lp, op, pp

        # 用于统计峰值内存以及训练时间
        self.Peak_Memory = 0
        self.PhaseStartTime = None
        self.PhaseEndTime = None

        # 训练开始前的准备工作
        self.TimeCost, self.Diary, self.EvaluateDiary, self.GaussianDiary, self.ImagesAlreadyBeTrainedIterations, self.source_path_list, self.model_path_list, self.IM = TrainingPreparation(self.args)

        # 用于在渐进式训练时记录此次加入的影像
        self.NewCams = []

        # 记录模型的已训练次数
        self.AlreadyTrainingIterations = 0
        self.StartFromTrainingIterations = 0

        # 在没有开始渐进式训练之前，影像匹配矩阵和影像权重不存在
        self.ImageMatchingMatrix = None
        self.Images_Weights = None

        # 在没有开始渐进式训练之前，新增影像数量不存在
        self.NewImagesNum = -1
        self.progressive_transition_index = 0
        self.CurrentGlobalOptimizationIterations = 0

        # 该字典用于存储每一张影像对应的高斯球
        self.Image_Visibility = {}
        self.MaxWeightsImages = []

        # 该列表用于存储一些影像的名称，仅在需要进行分块训练时会被用到
        # 若该列表不为空，则训练影像时只会从该列表中的影像来选取
        self.ChosenImageNames = []

        self.targetcam = None
        self.TDOM_Cam = None
        self.TDOM_alignment = None
        self.TDOM_pipeline = None
        self.targetcam_savetimes = 0
        self.TDOM_Cam_savetimes = 0

        # 用于存储是否需要进行GS可见度重新设置
        self.ResetOpacityTime = 0

        # 用于标记是否需要进行分块训练
        self.Drone_Image_Block = False
        self.Progressive_Training_Block = False

        # 表示当前模型被分为多少个部分进行了训练，初始为1，只有在内存不够时会创建新的块
        self.BlockNum = 1

        # 表示每一个块对应的pth文件被存储在了哪个位置
        self.BlockDirPath = []

        self.resolution_scales = [1.0]

        # 设置初始场景的文件路径
        self.args.model_path = self.model_path_list[0] if self.args.ContinueFromImageNo == -1 else self.model_path_list[self.args.ContinueFromImageNo - self.args.StartFromImageNo]
        self.args.source_path = self.source_path_list[0] if self.args.ContinueFromImageNo == -1 else self.source_path_list[self.args.ContinueFromImageNo - self.args.StartFromImageNo]
        self.Diary.write(f"First scene source_path: {self.args.source_path}\nFirst scene model_path: {self.args.model_path}\n")

    def _ensure_trainer_alive(self):
        """Fail promptly if the training child exited without a stop message."""
        supervisor = getattr(self, "worker_group", None)
        if supervisor is not None:
            supervisor.check()
            return
        for name, process in (
            ("GaussianTrainer", self.GaussianTrainer_process),
            ("output", getattr(self, "DataSaver_process", None)),
        ):
            if process is not None and not process.is_alive():
                process.join(timeout=0)
                raise RuntimeError(
                    f"{name} worker exited unexpectedly with code {process.exitcode}"
                )

    def _wait_for_trainer_message(self, timeout=0.1):
        """Block without busy-spinning while retaining child health checks."""
        while True:
            self._ensure_trainer_alive()
            try:
                return self.DataPreProccesser_queue.get(timeout=timeout)
            except Empty:
                continue

    # 为某一部分影像创建梯度图
    def GetGradLogForCams(self, cams=[], save=False):
        if cams == []:
            return 0

        for i in range(len(cams)):
            sys.stdout.write('\r')
            sys.stdout.write("Getting GradLog {}/{}".format(i + 1, len(cams)))
            sys.stdout.flush()

            if cams[i].GradLog == None:
                cams[i].GradLog, cams[i].MaxGradLog = Get_LogImage(cams[i].original_image, cams[i].image_name,
                                                                   os.path.join(self.args.Model_Path_Dir,
                                                                                "GradLogTrain"), save=save)

        sys.stdout.write('\n')

    def GetGradLogForImage(self, image, image_name, MG, save=False):
        return Get_LogImage(image, image_name, os.path.join(self.args.Model_Path_Dir, "GradLogRender"), MaxGradLog=MG, save=save)

    # 根据新的稀疏点云来扩张高斯点云
    def ExpandingGS_From_SparsePCD2(self, TrainCameras, get_ply=False):
        if os.path.exists(os.path.join(self.args.source_path_second, "sparse")):
            NewInputFormat = False
            CurrentImagesNames = list(TrainCameras.keys())
            train_cam_infos, test_cam_infos, nerf_normalization, point3D_ids_list, xys_list, tri_list = \
            sceneLoadTypeCallbacks[
                "Single_Image1"](
                self.args.source_path_second,
                self.args.images, self.args.eval,
                Do_Get_Tri_Mask=self.args.Use_Tri_Mask,
                Image_Name="",
                CurrentImagesNames=CurrentImagesNames,
                Diary=self.Diary,
                Silence=self.Silence,
                No_Key_Region=self.args.No_Key,
                NewInputFormat=NewInputFormat,
                Image_Folder_Path=self.args.Image_Folder_Path,
                retained_camera_mappings=TrainCameras)
        elif os.path.exists(os.path.join(self.args.source_path_second, "bin")):
            NewInputFormat = True
            CurrentImagesNames = list(TrainCameras.keys())
            train_cam_infos, test_cam_infos, nerf_normalization, point3D_ids_list, xys_list, tri_list = \
            sceneLoadTypeCallbacks[
                "Single_Image1"](
                self.args.source_path_second,
                self.args.images, self.args.eval,
                Do_Get_Tri_Mask=self.args.Use_Tri_Mask,
                Image_Name="",
                CurrentImagesNames=CurrentImagesNames,
                Diary=self.Diary,
                Silence=self.Silence,
                No_Key_Region=self.args.No_Key,
                NewInputFormat=NewInputFormat,
                Image_Folder_Path=self.args.Image_Folder_Path,
                retained_camera_mappings=TrainCameras)
        else:
            assert False, "Could not recognize scene type!"

        time1 = time.time()
        prepare_output_and_logger(self.args, True, Silence=self.Silence)

        # 初始化两个用于存储相机信息的数组：
        json_cams = []
        camlist = []

        # 如果测试影像或者训练影像存在，则将他们都加入到camlist中去
        if test_cam_infos:
            camlist.extend(test_cam_infos)
        if train_cam_infos:
            camlist.extend(train_cam_infos)

        # 将camlist中的影像信息转换为JSON的模式，并存储在json_cams中
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))

        # 将json_cams中的数据存在cameras.json中
        with open(os.path.join(self.args.model_path_second, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        # 将包裹所有相机的球的半径赋值给self.cameras_extent（相机分布范围）
        if self.args.spatial_lr_scale == 0.0:
            # self.scene.cameras_extent = nerf_normalization["radius"]
            cameras_extent = nerf_normalization["radius"]
        else:
            # self.scene.cameras_extent = self.args.spatial_lr_scale
            cameras_extent = self.args.spatial_lr_scale

        time2 = time.time()
        # print(f"Before Add New Images: {time2 - time1}s")
        self.Diary.write(f"Before Add New Images: {time2 - time1}s\n")

        # 将新加入的影像读入到系统之中
        New_scene_info_traincam, NewCams, UpdatedTrainCamsPos = self.AddNewImages2(train_cam_infos, TrainCameras, threshold=self.args.LogMaskThreshold)

        scene_info = sceneLoadTypeCallbacks["Single_Image2"](self.args.source_path_second,
                                                             train_cam_infos, test_cam_infos, nerf_normalization,
                                                             point3D_ids_list, xys_list, tri_list, NewCams,
                                                             self.args.OriginImageHeight, self.args.OriginImageWidth,
                                                             get_ply=get_ply,
                                                             points_per_triangle=self.args.points_per_triangle,
                                                             device="cuda", Diary=self.Diary, Silence=self.Silence, NewInputFormat=NewInputFormat,
                                                             Ply_Cache_Path=os.path.join(
                                                                 self.args.dataset_cache_root,
                                                                 "point_cloud",
                                                                 os.path.basename(self.args.source_path_second),
                                                                 "points3D.ply",
                                                             ))

        # Single_Image2 has consumed these buffers while sampling the new sparse
        # points. They are not inputs to rendering or gradient computation.
        release_camera_sampling_buffers(NewCams)

        time1 = time.time()
        if scene_info.ply_path is not None:
            with open(scene_info.ply_path, 'rb') as src_file, open(
                    os.path.join(self.args.model_path_second, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())

        time2 = time.time()
        if not self.Silence:
            print(f"Copy Ply File: {time2 - time1}s")
        self.Diary.write(f"Copy Ply File: {time2 - time1}s\n")
        '''
        # 对于当前新影像，重置该影像内全部高斯球的可见度，然后进行一次剔除
        for newcam in NewCams:
            render_pkg = render(newcam, self.gaussians, self.args, self.bg, Render_Mask=newcam.Tri_Mask)'''

        # # 根据扩张后的稀疏点云来更新高斯点云
        # time1 = time.time()
        # self.gaussians.expand_from_pcd(scene_info.point_cloud, self.scene.cameras_extent)
        # time2 = time.time()
        # print(f"expand_from_pcd: {time2 - time1}s")
        # self.Diary.write(f"expand_from_pcd: {time2 - time1}s\n")
        #
        # # 记录这一次的稀疏点云
        # self.scene.basic_pcd = scene_info.point_cloud
        #
        # # 更新模型输出位置
        # self.scene.model_path = self.args.model_path_second

        return New_scene_info_traincam, NewCams, UpdatedTrainCamsPos, scene_info, self.args.model_path_second, cameras_extent

    # 读入新的影像数据并更新所有已有影像的位姿信息
    def AddNewImages2(self, train_cam_infos, TrainCameras, threshold=0.1):
        time1 = time.time()

        # 为self.train_cameras进行更新，将新加入的影像的相关信息加入到其中
        # 先找到新加入的影像
        NEW_TrainCamInfos = train_cam_infos
        CurrentNewImagesNum = 0
        NewImageNames = "["
        NewCamInfos = []
        for i in range(len(NEW_TrainCamInfos)):
            if NEW_TrainCamInfos[i].image is not None:
                New_CamInfo = NEW_TrainCamInfos[i]
                NewImageNames = NewImageNames + New_CamInfo.image_name + ", "
                NewCamInfos.append(New_CamInfo)
                CurrentNewImagesNum = CurrentNewImagesNum + 1

        nominal_new_images = self.NewImagesNum
        self.NewImagesNum = CurrentNewImagesNum
        if nominal_new_images != CurrentNewImagesNum:
            message = (
                "Progressive stage-number delta ({}) differs from the actual "
                "new-image count ({}); using the camera-set difference."
            ).format(nominal_new_images, CurrentNewImagesNum)
            print(message)
            self.Diary.write(message + "\n")

        NewImageNames = NewImageNames + "]"
        if not self.Silence:
            print(f"Newly Added Images Name: {NewImageNames}")
        self.Diary.write(f"Newly Added Images Name: {NewImageNames}\n")

        # 在每一个resolution_scale下：
        NewCams = []
        UpdatedTrainCamsPos = {}
        for resolution_scale in self.resolution_scales:
            # 更新self.train_cameras中[resolution_scale]每一个影像的位姿信息
            # for i in range(len(self.scene.train_cameras[resolution_scale])):
            for cam in NEW_TrainCamInfos:
                if (cam.image_name in TrainCameras):
                    world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, TrainCameras[cam.image_name]["trans"], TrainCameras[cam.image_name]["scale"])).transpose(0, 1) # 4*4
                    projection_matrix = getProjectionMatrix(znear=TrainCameras[cam.image_name]["znear"],
                                                            zfar=TrainCameras[cam.image_name]["zfar"],
                                                            fovX=cam.FovX,
                                                            fovY=cam.FovY).transpose(0, 1) # 4*4
                    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                    camera_center = world_view_transform.inverse()[3, :3]

                    replacement = None
                    intrinsics_changed = camera_image_mapping_changed(
                        cam, TrainCameras[cam.image_name]
                    )
                    if intrinsics_changed:
                        image = load_rectified_image(cam.image_path, cam.calibration)
                        resized = PILtoTorch(
                            image,
                            (
                                TrainCameras[cam.image_name]["image_width"],
                                TrainCameras[cam.image_name]["image_height"],
                            ),
                        )
                        replacement = Camera(
                            colmap_id=cam.uid,
                            R=cam.R,
                            T=cam.T,
                            FoVx=cam.FovX,
                            FoVy=cam.FovY,
                            image=resized[:3, ...],
                            gt_alpha_mask=(resized[3:4, ...] if resized.shape[0] == 4 else None),
                            Tri_Mask=resize_camera_mask(
                                cam.Tri_Mask,
                                (
                                    TrainCameras[cam.image_name]["image_width"],
                                    TrainCameras[cam.image_name]["image_height"],
                                ),
                            ),
                            image_name=cam.image_name,
                            uid=TrainCameras[cam.image_name]["uid"],
                            data_device=self.args.data_device,
                            image_path=cam.image_path,
                            calibration=cam.calibration,
                        )

                    UpdatedTrainCamsPos[cam.image_name] = [
                        world_view_transform,
                        projection_matrix,
                        full_proj_transform,
                        camera_center,
                        cam,
                        replacement,
                    ]

            # for i in range(len(TrainCameras)):
            #     for cam in NEW_TrainCamInfos:
            #         if (cam.image_name == TrainCameras[i].image_name):
            #             # TrainCameras[i].R = cam.R
            #             # TrainCameras[i].T = cam.T
            #         # if (cam.image_name == self.scene.train_cameras[resolution_scale][i].image_name):
            #         #     self.scene.train_cameras[resolution_scale][i].R = cam.R
            #         #     self.scene.train_cameras[resolution_scale][i].T = cam.T
            #
            #
            #             world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, TrainCameras[i].trans, TrainCameras[i].scale)).transpose(0, 1)
            #             projection_matrix = getProjectionMatrix(znear=TrainCameras[i].znear, zfar=TrainCameras[i].zfar, fovX=TrainCameras[i].FoVx, fovY=TrainCameras[i].FoVy).transpose(0, 1)
            #             full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            #             camera_center = world_view_transform.inverse()[3, :3]
            #
            #             UpdatedTrainCamsPos[TrainCameras[i].image_name] = [world_view_transform, projection_matrix, full_proj_transform, camera_center]
            #
            #             break

            for cam_info in NewCamInfos:
                NewCam = loadCam(self.args, len(TrainCameras), cam_info, resolution_scale)

                self.GetGradLogForCams([NewCam])

                with torch.no_grad():
                    # render_pkg = render(NewCam, self.gaussians, self.args, self.bg, Render_Mask=NewCam.Tri_Mask)
                    # image = render_pkg["render"]

                    # 像后端发送指令，令其渲染一张当前新影像
                    msg = ["Render_New", NewCam]
                    self.GaussianTrainer_queue.put(msg)

                    image = self._wait_for_trainer_message()[0]

                    renderGradLog, _ = self.GetGradLogForImage(image, NewCam.image_name, NewCam.MaxGradLog)

                    Log_minus = LogImage_minus(NewCam.GradLog, renderGradLog,
                                               os.path.join(self.args.Model_Path_Dir, "GradLogMinus"),
                                               NewCam.image_name)

                    NewCam.LogMask = Log_minus >= threshold

                NewCams.append(NewCam)

                if self.args.Block or self.ChosenImageNames != []:
                    self.ChosenImageNames.append(NewCam.image_name)

            msg = ["RenderNewDone"]
            self.GaussianTrainer_queue.put(msg)

            # self.scene.scene_info_traincam = train_cam_infos
            New_scene_info_traincam = train_cam_infos

            time2 = time.time()
            # print(f"Add New Images: {time2 - time1}s")
            self.Diary.write(f"Add New Images: {time2 - time1}s\n")

            return New_scene_info_traincam, NewCams, UpdatedTrainCamsPos

    # 该函数将被送入到一个线程中，用于渐进式训练中的数据读取以及预处理
    def run(self):
        if self.args.start_checkpoint is not None:
            checkpoint = load_training_checkpoint(
                self.args.start_checkpoint, map_location="cpu"
            )
            restore_rng_state(checkpoint["data_preprocessor_rng_state"])
        self.GaussianTrainer_queue.put(["init"])

        StartFrom = 0 if self.args.ContinueFromImageNo == -1 else self.args.ContinueFromImageNo - self.args.StartFromImageNo
        if StartFrom > len(self.source_path_list) - 2:
            while True:
                command = self._wait_for_trainer_message()
                if command[0] in ("Progressive_First_Step", "Progressive"):
                    self.args.source_path = self.source_path_list[-1]
                    self.args.model_path = self.model_path_list[-1]
                    self.args.source_path_second = self.source_path_list[-1]
                    self.args.model_path_second = self.model_path_list[-1]
                    print("DataPreProccesser: Start Final Refinement...")
                    self.GaussianTrainer_queue.put(
                        ["Final_Refinement", self.args.model_path]
                    )
                    while True:
                        if self._wait_for_trainer_message()[0] == "stop":
                            return 0
                if command[0] == "stop":
                    return 0

        ProgressiveTrainingTime = StartFrom
        while True:
            Commond_from_GaussianTrainer = self._wait_for_trainer_message()
            if Commond_from_GaussianTrainer is not None:
                if Commond_from_GaussianTrainer[0] == "Progressive_First_Step":
                    if (ProgressiveTrainingTime < len(self.source_path_list) - 1) and (ProgressiveTrainingTime + self.args.StartFromImageNo + 1 <= self.args.EndAtImageNo or self.args.EndAtImageNo == -1):
                        # 数据更新同时计算这一次有多少新增影像
                        self.NewImagesNum = self.IM[ProgressiveTrainingTime + 1] - self.IM[ProgressiveTrainingTime]
                        self.args.source_path = self.source_path_list[ProgressiveTrainingTime]
                        self.args.model_path = self.model_path_list[ProgressiveTrainingTime]
                        self.args.source_path_second = self.source_path_list[ProgressiveTrainingTime + 1]
                        self.args.model_path_second = self.model_path_list[ProgressiveTrainingTime + 1]

                        # 根据新的稀疏点云来扩张高斯点云
                        get_ply = True if ProgressiveTrainingTime == len(self.source_path_list) - 2 else False
                        # self.NewCams = self.ExpandingGS_From_SparsePCD(get_ply=get_ply)
                        New_scene_info_traincam, NewCams, UpdatedTrainCamsPos, scene_info, new_model_path_second, cameras_extent = self.ExpandingGS_From_SparsePCD2(Commond_from_GaussianTrainer[1], get_ply=get_ply)
                        # print(f"ChosenImagesNum: {len(self.ChosenImageNames)}")

                        names = list(UpdatedTrainCamsPos.keys())
                        worlds = torch.stack([UpdatedTrainCamsPos[n][0] for n in names])  # (N,4,4)
                        projs = torch.stack([UpdatedTrainCamsPos[n][1] for n in names])  # (N,4,4)
                        fulls = torch.stack([UpdatedTrainCamsPos[n][2] for n in names])  # (N,4,4)
                        centers = torch.stack([UpdatedTrainCamsPos[n][3] for n in names])  # (N,3)
                        camera_infos = [UpdatedTrainCamsPos[n][4] for n in names]
                        replacements = [UpdatedTrainCamsPos[n][5] for n in names]

                        # 像后端发送指令和数据，开始对新进影像的训练
                        print("DataPreProccesser: Send Information to GaussianTrainer...")
                        # msg = ["Progressive_Train", New_scene_info_traincam, NewCams, TrainCameras, scene_info, new_model_path_second, cameras_extent, self.NewImagesNum]
                        msg = ["Progressive_Train", None, NewCams, [worlds, projs, fulls, centers, names, camera_infos, replacements], scene_info.point_cloud, new_model_path_second, cameras_extent, self.NewImagesNum, capture_rng_state()]
                        self.GaussianTrainer_queue.put(msg)

                        ProgressiveTrainingTime += 1
                elif Commond_from_GaussianTrainer[0] == "Progressive":
                    if (ProgressiveTrainingTime + self.args.StartFromImageNo + 1 <= self.args.EndAtImageNo or self.args.EndAtImageNo == -1) and (ProgressiveTrainingTime < len(self.source_path_list) - 1):
                        # 数据更新同时计算这一次有多少新增影像
                        self.NewImagesNum = self.IM[ProgressiveTrainingTime + 1] - self.IM[ProgressiveTrainingTime]
                        self.args.source_path = self.source_path_list[ProgressiveTrainingTime]
                        self.args.model_path = self.model_path_list[ProgressiveTrainingTime]
                        self.args.source_path_second = self.source_path_list[ProgressiveTrainingTime + 1]
                        self.args.model_path_second = self.model_path_list[ProgressiveTrainingTime + 1]

                        # 根据新的稀疏点云来扩张高斯点云
                        get_ply = True if ProgressiveTrainingTime == len(self.source_path_list) - 2 else False
                        # self.NewCams = self.ExpandingGS_From_SparsePCD(get_ply=get_ply)
                        New_scene_info_traincam, NewCams, UpdatedTrainCamsPos, scene_info, new_model_path_second, cameras_extent = self.ExpandingGS_From_SparsePCD2(Commond_from_GaussianTrainer[1], get_ply=get_ply)
                        # print(f"ChosenImagesNum: {len(self.ChosenImageNames)}")

                        names = list(UpdatedTrainCamsPos.keys())
                        worlds = torch.stack([UpdatedTrainCamsPos[n][0] for n in names])  # (N,4,4)
                        projs = torch.stack([UpdatedTrainCamsPos[n][1] for n in names])  # (N,4,4)
                        fulls = torch.stack([UpdatedTrainCamsPos[n][2] for n in names])  # (N,4,4)
                        centers = torch.stack([UpdatedTrainCamsPos[n][3] for n in names])  # (N,3)
                        camera_infos = [UpdatedTrainCamsPos[n][4] for n in names]
                        replacements = [UpdatedTrainCamsPos[n][5] for n in names]

                        # 像后端发送指令和数据，开始对新进影像的训练，此时，需要注意先等后端把上一张影像完成训练之后，再进行传输
                        if self._wait_for_trainer_message()[0] == "Progressive_finish":
                            # 像后端发送指令和数据，开始对新进影像的训练
                            print("DataPreProccesser: Send Information to GaussianTrainer...")
                            msg = ["Progressive_Train", None, NewCams, [worlds, projs, fulls, centers, names, camera_infos, replacements], scene_info.point_cloud, new_model_path_second, cameras_extent, self.NewImagesNum, capture_rng_state()]
                            # msg = ["Progressive_Train"]
                            self.GaussianTrainer_queue.put(msg)

                        ProgressiveTrainingTime += 1
                    else:
                        msg = ["Final_Refinement_Prepare"]
                        self.GaussianTrainer_queue.put(msg)

                        # 像后端发送指令和数据，开始进行最终的全局优化，此时，需要注意先等后端把上一张影像完成训练之后，再进行传输
                        while True:
                            if self._wait_for_trainer_message()[0] == "Progressive_finish":
                                print("*****************************************************************************")
                                print("DataPreProccesser: Start Final Refinement...")
                                # The final source stage was already expanded and
                                # trained. Reuse its output directory instead of
                                # reserving (and thereby skipping) the last stage.
                                self.args.source_path = self.source_path_list[-1]
                                self.args.model_path = self.model_path_list[-1]
                                self.args.source_path_second = self.source_path_list[-1]
                                self.args.model_path_second = self.model_path_list[-1]

                                msg = ["Final_Refinement", self.args.model_path]
                                self.GaussianTrainer_queue.put(msg)

                                break
                elif Commond_from_GaussianTrainer[0] == "stop":
                    break

# 后端，第二个线程，主要用于高斯训练
class GaussianTrainer(mp.Process):
    # 构造函数
    def __init__(self, args=None, lp=None, op=None, pp=None):
        super().__init__()

        # 获取参数配置器
        if args is None:
            self.args, self.lp, self.op, self.pp = GetArgs()
        else:
            self.args = copy.deepcopy(args)
            self.lp, self.op, self.pp = lp, op, pp

        self.ImagesAlreadyBeTrainedIterations = None
        self.ResetOpacityTime = 0

        # 用于两个进程之间的信息传递
        self.DataPreProccesser_queue = None
        self.GaussianTrainer_queue = None
        self.DataSaver_queue = None

        # 记录模型的已训练次数
        self.AlreadyTrainingIterations = 0
        self.StartFromTrainingIterations = 0
        self.progressive_transition_index = 0
        self.CurrentGlobalOptimizationIterations = 0
        self.data_preprocessor_rng_state = None
        self.training_progress_stages = set(
            select_training_progress_stages(
                self.args.StartFromImageNo,
                self.args.EndAtImageNo,
                self.args.training_progress_target_frames,
            )
        )
        self.last_semantic_sync_result = None

        self.Drone_Image_Block = False
        self.Progressive_Training_Block = False

        self.source_path_list, self.model_path_list = None, None
        self.resolution_scales = [1.0]

        # 该字典用于存储每一张影像对应的高斯球
        self.Image_Visibility = {}
        self.MaxWeightsImages = []

        # 该列表用于存储一些影像的名称，仅在需要进行分块训练时会被用到
        # 若该列表不为空，则训练影像时只会从该列表中的影像来选取
        self.ChosenImageNames = []

        # TDOM的地面分辨率
        self.GSD_x = self.args.GSD_x
        self.GSD_y = self.args.GSD_y

        self.NewCams = []

        self.commond = None

        self.imagenum = self.args.StartFromImageNo

        self.TDOM_Cam = None
        self.TDOM_alignment = None
        self.TDOM_pipeline = None
        self.TDOM_grid = None

        self.TimeCost = {
            "GaussianInitial": 0.0,
            "ProgressiveTrain": 0.0,
            "ProgressiveEndToEnd": 0.0,
            "FinalRefinement": 0.0,
        }
        self.ProgressiveStageTimings = []

        self.EvaluateDiary = None

        self.Finish = False

        # 在SLAM数据集上可能出现训练速度快于前端处理速度的现象，因此该布尔值用于表示是否已经完成“Render_New”这个操作
        self.RenderNewDone = False

        # Optional WP3 sidecar. It is deliberately outside every RGB optimizer.
        self.semantic_hook = None
        self.semantic_tdom_config = None
        self.semantic_planar_fallback_config = None
        self.semantic_publication_policy = None
        self.semantic_map_error = None
        if self.args.enable_semantics:
            try:
                self.semantic_publication_policy = MapPublicationPolicy(
                    publication_stage_interval=(
                        self.args.semantic_map_publication_interval
                    ),
                    checkpoint_stage_interval=(
                        self.args.semantic_checkpoint_interval
                    ),
                    min_free_cuda_bytes=(
                        self.args.semantic_tdom_min_free_cuda_bytes
                    ),
                    max_pair_output_bytes=(
                        self.args.semantic_tdom_max_output_bytes
                    ),
                    online_time_budget_seconds=(
                        self.args.semantic_online_time_budget_seconds
                    ),
                    cuda_stream_priority=(
                        self.args.semantic_cuda_stream_priority
                    ),
                )
            except Exception as error:
                self.semantic_map_error = str(error)
                SemanticWarning(
                    "semantic publication policy is invalid; RGB geometry "
                    "continues: {}".format(error)
                )
        grid_path = os.path.join(
            self.args.map_output_dir, "grid.json"
        )
        if os.path.isfile(grid_path):
            try:
                self.TDOM_grid = MapGridSpec.load(grid_path)
            except Exception as error:
                self.semantic_map_error = str(error)
                SemanticWarning(
                    "fixed map grid is invalid; RGB geometry continues: {}".format(
                        error
                    )
                )

    def _initialize_semantic_hook(self, force_fresh=False):
        if not self.args.enable_semantics:
            return
        try:
            semantic_dataset_root = self.args.semantic_data_root
            _, render_palette, render_palette_source = (
                resolve_semantic_render_palette(
                    semantic_dataset_root,
                    self.args.semantic_tdom_palette_path,
                )
            )
            config = OnlineSemanticConfig(
                overlap_threshold=self.args.semantic_overlap_threshold,
                max_group_size=self.args.semantic_max_group_size,
                reliability_floor=self.args.semantic_reliability_floor,
                temperature=self.args.semantic_temperature,
                boundary_bandwidth=self.args.semantic_boundary_bandwidth,
                boundary_floor=self.args.semantic_boundary_floor,
            )
            self.semantic_tdom_config = SemanticTdomConfig(
                min_coverage=self.args.semantic_tdom_min_coverage,
                min_absolute_support=(
                    self.args.semantic_tdom_min_absolute_support
                ),
                min_support_ratio=self.args.semantic_tdom_min_support_ratio,
                render_tile_rows=(
                    self.args.semantic_tdom_render_tile_rows
                ),
                finalize_tile_rows=(
                    self.args.semantic_tdom_finalize_tile_rows
                ),
                save_diagnostics=(
                    self.args.semantic_tdom_save_diagnostics
                ),
                save_probabilities=(
                    self.args.semantic_tdom_save_probabilities
                ),
                max_output_bytes=self.args.semantic_tdom_max_output_bytes,
                min_free_cuda_bytes=(
                    self.args.semantic_tdom_min_free_cuda_bytes
                ),
                max_gaussian_scale_ratio=(
                    self.args.semantic_tdom_max_gaussian_scale_ratio
                ),
                render_palette=render_palette,
                render_palette_source=render_palette_source,
            )
            self.semantic_planar_fallback_config = PlanarFallbackConfig(
                class_name=(
                    self.args.semantic_tdom_planar_fallback_class.strip()
                ),
                min_component_pixels=(
                    self.args.semantic_tdom_planar_fallback_min_component_pixels
                ),
                ring_radius_pixels=(
                    self.args.semantic_tdom_planar_fallback_ring_radius_pixels
                ),
                min_boundary_consensus=(
                    self.args.semantic_tdom_planar_fallback_min_boundary_consensus
                ),
                plane_context_pixels=(
                    self.args.semantic_tdom_planar_fallback_plane_context_pixels
                ),
                min_plane_gaussians=(
                    self.args.semantic_tdom_planar_fallback_min_plane_gaussians
                ),
                max_plane_mad_world=(
                    self.args.semantic_tdom_planar_fallback_max_plane_mad_world
                ),
                min_views=(
                    self.args.semantic_tdom_planar_fallback_min_views
                ),
                min_agreement=(
                    self.args.semantic_tdom_planar_fallback_min_agreement
                ),
                min_groups=(
                    self.args.semantic_tdom_planar_fallback_min_groups
                ),
            )
            self.semantic_hook = AsyncProgressiveSemanticHook.create(
                dataset_root=self.args.Source_Path_Dir,
                output_root=os.path.join(
                    self.args.semantic_work_dir
                ),
                gaussians=self.gaussians,
                config=config,
                semantic_dataset_root=semantic_dataset_root,
                publication_policy=self.semantic_publication_policy,
                resume=not force_fresh,
                final_authoritative_complete=(
                    self.args.semantic_authoritative_exact_complete
                ),
                preflight_report_path=getattr(
                    self.args, "dataset_preflight_path", None
                ),
                preflight_report_sha256=getattr(
                    self.args, "dataset_preflight_sha256", None
                ),
            )
            self.gaussians.set_topology_observer(
                self.semantic_hook.apply_topology, owner=self.semantic_hook
            )
        except Exception as error:
            self.semantic_hook = None
            SemanticWarning(
                "semantic initialization failed; RGB geometry continues: {}".format(
                    error
                )
            )

    def _run_semantic_hook(self, stage_name, final=False):
        if self.semantic_hook is None:
            return None
        hook = self.semantic_hook
        snapshot_root = os.path.join(
            self.args.Source_Path_Dir, str(self.imagenum)
        )
        expected_training_type = {
            "initialization": "Initialization",
            "progressive": "On_The_Fly",
            "final_refinement": "Final_Refinement",
        }.get(stage_name)
        alignment = self.TDOM_alignment or {}
        render_semantic_tdom = (
            stage_name in ("initialization", "progressive", "final_refinement")
            and self.args.render_TDOM
            and (not self.args.semantic_tdom_final_only or final)
            and self.TDOM_Cam is not None
            and alignment.get("geometry_stage") == int(self.imagenum)
            and alignment.get("training_type") == expected_training_type
        )
        publication_request = None
        try:
            if render_semantic_tdom:
                pair_output_root = os.path.join(
                    self.args.map_output_dir,
                    "releases",
                    alignment["product_key"],
                )
                publication_request = MapPublicationRequest(
                    product_key=alignment["product_key"],
                    output_root=pair_output_root,
                    camera=self.TDOM_Cam,
                    alignment=self.TDOM_alignment,
                    rgb_renderer=render,
                    rgb_pipeline=self.TDOM_pipeline,
                    rgb_background=self.background,
                    semantic_config=self.semantic_tdom_config,
                    planar_fallback_config=(
                        self.semantic_planar_fallback_config
                    ),
                    publication_due=(
                        final
                        or self.semantic_publication_policy.should_publish(
                            self.imagenum, final
                        )
                    ),
                )
        except Exception as error:
            SemanticWarning(
                "map publication request is invalid; semantic fusion "
                "continues: {}".format(error)
            )
        capture = None
        capture_started = time.perf_counter()
        if not final and self._training_progress_due(expected_training_type):
            capture = self._make_training_progress_capture(expected_training_type)
        try:
            result = hook.synchronize(
                geometry_stage=self.imagenum,
                cameras=self.scene.getTrainCameras(),
                gaussians=self.gaussians,
                snapshot_root=snapshot_root,
                final=final,
                publication_request=publication_request,
                on_synchronized=capture,
            )
            if capture is not None:
                self._queued_progress_capture = (
                    (int(self.imagenum), expected_training_type),
                    dict(requested=True, asynchronous=True, semantic_wait_seconds=0.0,
                         rgb_tdom_seconds=0.0, semantic_tdom_seconds=0.0,
                         frame_enqueue_seconds=0.0,
                         capture_submit_seconds=time.perf_counter()-capture_started,
                         capture_backpressure_seconds=hook.last_capture_backpressure_seconds))
            self.last_semantic_sync_result = result
            print(
                "Semantic sync [{}]: geometry={}, through={}, lag={}, map={}".format(
                    stage_name,
                    result.geometry_stage,
                    result.semantic_through_image,
                    result.lag_batches,
                    result.publication_status,
                )
            )
            if result.errors:
                SemanticWarning(
                    "semantic sync deferred after error; RGB geometry continues: {}".format(
                        result.errors[0]
                    )
                )
            if not result.within_lag_budget:
                SemanticWarning(
                    "semantic output is more than one progressive batch behind"
                )
            if final:
                self.gaussians.set_topology_observer(None)
                hook.close()
                prepared = getattr(hook.core.projector, "prepared_labels", None)
                write_json_atomic(Path(self.args.metrics_dir) / "semantic_pipeline.json",
                    dict(schema_version=1, captures=dict(hook.capture_stats),
                         prepared_labels=None if prepared is None else dict(prepared.stats)))
            return result
        except Exception as error:
            SemanticWarning(
                "semantic stage hook failed; RGB geometry continues: {}".format(
                    error
                )
            )
            hook.close(reason=error, wait=bool(final))
            self.gaussians.set_topology_observer(None)
            self.semantic_hook = None
            self.last_semantic_sync_result = None
            return None

    def InitialGaussianTrain(self):
        # 设置初始场景的文件路径
        self.args.model_path = self.model_path_list[0] if self.args.ContinueFromImageNo == -1 else self.model_path_list[self.args.ContinueFromImageNo - self.args.StartFromImageNo]
        self.args.source_path = self.source_path_list[0] if self.args.ContinueFromImageNo == -1 else self.source_path_list[self.args.ContinueFromImageNo - self.args.StartFromImageNo]
        self.args.point_cloud_cache_path = os.path.join(
            self.args.dataset_cache_root,
            "point_cloud",
            os.path.basename(self.args.source_path),
            "points3D.ply",
        )
        # self.Diary.write(f"First scene source_path: {self.args.source_path}\nFirst scene model_path: {self.args.model_path}\n")

        # 初始化背景颜色
        self.bg_color = [1, 1, 1] if self.args.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.bg = torch.rand((3), device="cuda") if self.args.random_background else self.background

        # 初始化log损失累计
        self.ema_loss_for_log = 0.0

        # 主要进行一系列的模型以及其它文件的输出准备（例如构建输出文件夹等），并返回一个Tensorboard writer对象
        self.tb_writer = prepare_output_and_logger(self.args)

        # 初始化3D Gaussian Splatting模型，主要是一些模型属性的初始化和神经网络中一些激活函数的初始化
        self.gaussians = GaussianModel(self.args.sh_degree)

        # 初始化一个场景
        self.scene = Scene(self.args, self.gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0], Do_Get_Tri_Mask=self.args.Use_Tri_Mask, No_Key_Region=self.args.No_Key)

        # 进行一些训练上的初始化设置
        self.gaussians.training_setup(self.args)
        if self.args.start_checkpoint is not None:
            checkpoint = load_training_checkpoint(
                self.args.start_checkpoint, map_location="cuda"
            )
            if checkpoint["phase"] == "run_complete":
                raise ValueError("checkpoint already represents a completed run")
            expected_preflight = getattr(
                self.args, "dataset_preflight_sha256", None
            )
            if checkpoint.get("dataset_preflight_sha256") != expected_preflight:
                raise ValueError(
                    "checkpoint dataset preflight does not match this run"
                )
            if checkpoint.get("source_path_dir") != str(
                Path(self.args.Source_Path_Dir).resolve()
            ):
                raise ValueError("checkpoint progressive dataset root does not match")
            self.gaussians.restore(checkpoint["gaussians"], self.args)
            restored_cameras = [
                camera_from_state(
                    item, data_device=self.args.data_device,
                    expected_resolution=self.args.resolution,
                )
                for item in checkpoint["cameras"]
            ]
            self.scene.setTrainCameras(restored_cameras)
            self.scene.cameras_extent = float(checkpoint["cameras_extent"])
            self.ImagesAlreadyBeTrainedIterations.clear()
            self.ImagesAlreadyBeTrainedIterations.update(
                checkpoint["images_already_trained_iterations"]
            )
            self.AlreadyTrainingIterations = int(checkpoint["iteration"])
            self.StartFromTrainingIterations = self.AlreadyTrainingIterations
            self.progressive_transition_index = int(
                checkpoint["progressive_transition_index"]
            )
            self.ResetOpacityTime = int(checkpoint.get("reset_opacity_time", 0))
            self.imagenum = int(checkpoint["stage"])
            restore_rng_state(checkpoint["rng_state"])

        # 进行初始模型的训练
        time1 = time.time()
        if (not self.args.skip_FirstSceneTraining) and self.args.start_checkpoint is None:
            self.Train_Gaussians(self.args.IterationFirstScene, "Initialization")
        time2 = time.time()
        self.TimeCost["GaussianInitial"] = time2 - time1

        # WP3 synchronization point: initialization/densify/prune is complete.
        if self.args.render_TDOM and self.TDOM_Cam is None:
            with torch.no_grad():
                self._prepare_semantic_map_safely("Initialization")
                if not self.args.enable_semantics:
                    self._render_legacy_tdom()
        if semantic_online_required(self.args):
            self._initialize_semantic_hook()
        self._run_semantic_hook("initialization")
        with torch.no_grad():
            self._capture_training_progress_tdom("Initialization")
        if self.args.render_training_progress and self.semantic_hook is not None:
            self.semantic_hook.wait_for_idle()

        # 向前端传递信息，表示初始训练已经完成，可以开始读取后续的影像
        self.SendImagePos_to_DataPreProccesser(True)

    # 为某一部分影像创建梯度图
    def GetGradLogForCams(self, cams=[], save=False):
        if cams == []:
            cams = self.scene.getTrainCameras()

        for i in range(len(cams)):
            sys.stdout.write('\r')
            sys.stdout.write("Getting GradLog {}/{}".format(i + 1, len(cams)))
            sys.stdout.flush()

            if cams[i].GradLog == None:
                cams[i].GradLog, cams[i].MaxGradLog = Get_LogImage(cams[i].original_image, cams[i].image_name, os.path.join(self.args.Model_Path_Dir, "GradLogTrain"), save=save)

        sys.stdout.write('\n')

    def ImagesAlreadyBeTrainedIterations_Set(self):
        for NewCam in self.NewCams:
            Equivalent_training_times = 0
            self.ImagesAlreadyBeTrainedIterations[NewCam.image_name] = Equivalent_training_times
            # print(f"{NewCam.image_name}: Equavelent Training Times = {Equivalent_training_times}")

    # 将现有影像的位姿信息传给前端
    def SendImagePos_to_DataPreProccesser(self, IsFirstStep=False):
        TrainCamsDict = {}
        for cam in self.scene.getTrainCameras():
            TrainCamsDict[cam.image_name] = {"trans": cam.trans,
                                             "scale": cam.scale,
                                             "znear": cam.znear,
                                             "zfar": cam.zfar,
                                             "FoVx": cam.FoVx,
                                             "FoVy": cam.FoVy,
                                             "calibration": cam.calibration,
                                             "uid": cam.uid,
                                             "image_width": cam.image_width,
                                             "image_height": cam.image_height}

        # 向前端传递信息，表示开始读取后续的影像
        if IsFirstStep:
            msg = ["Progressive_First_Step", TrainCamsDict]
            self.DataPreProccesser_queue.put(msg)
        else:
            msg = ["Progressive", TrainCamsDict]
            self.DataPreProccesser_queue.put(msg)

    def _prepare_tdom_publication(self, training_type):
        """Create one fixed-grid map event without rendering either product."""
        if not self.args.render_TDOM:
            return
        final = training_type == "Final_Refinement"
        progress_due = self._training_progress_due(training_type, final=final)
        if self.args.enable_semantics and (
            self.semantic_publication_policy is None
            or (
                self.args.semantic_tdom_final_only
                and not final
                and not progress_due
            )
        ):
            return

        cameras = self.scene.getTrainCameras()
        if not cameras:
            raise RuntimeError("TDOM publication requires at least one camera")
        reference_camera = min(cameras, key=lambda item: str(item.image_name))
        points = self.gaussians.get_xyz

        if self.TDOM_grid is None:
            reference_rotation = torch.tensor(reference_camera.R)
            reference_translation = torch.tensor(reference_camera.T)
            rectangle = camera_orthogonal_bounding_rectangle(
                points, reference_rotation, reference_translation
            )
            ymin, ymax, xmin, xmax = rectangle["xy_max_min"]
            normal = rectangle["normal"].detach().cpu().tolist()
            # The historical TDOM camera maps its image columns to v and rows
            # to u.  Make that formerly implicit axis swap part of the grid.
            pixel_x = rectangle["v"].detach().cpu().tolist()
            pixel_y = rectangle["u"].detach().cpu().tolist()
            plane_offset = float(
                torch.median(points @ rectangle["normal"]).item()
            )
            self.TDOM_grid = MapGridSpec.create(
                plane_normal_world=normal,
                pixel_x_axis_world=pixel_x,
                pixel_y_axis_world=pixel_y,
                plane_offset_world=plane_offset,
                first_bounds=(ymin, ymax, xmin, xmax),
                requested_gsd=(self.GSD_x, self.GSD_y),
                tile_shape=(
                    self.args.semantic_map_tile_size,
                    self.args.semantic_map_tile_size,
                ),
            )
            self.TDOM_grid.save(
                os.path.join(
                    self.args.map_output_dir,
                    "grid.json",
                )
            )

        grid = self.TDOM_grid
        pixel_x = torch.tensor(
            grid.pixel_x_axis_world, device=points.device, dtype=points.dtype
        )
        pixel_y = torch.tensor(
            grid.pixel_y_axis_world, device=points.device, dtype=points.dtype
        )
        projected_x = points @ pixel_x
        projected_y = points @ pixel_y
        inlier = remove_outliers_quantile(projected_x, projected_y, q=0.9995)
        projected_x = projected_x[inlier]
        projected_y = projected_y[inlier]
        bounds = (
            float(projected_x.min().item()),
            float(projected_x.max().item()),
            float(projected_y.min().item()),
            float(projected_y.max().item()),
        )
        window = grid.window_for_bounds(bounds)

        normal = np.asarray(grid.plane_normal_world, dtype=np.float64)
        image_x = np.asarray(grid.pixel_x_axis_world, dtype=np.float64)
        image_y = np.asarray(grid.pixel_y_axis_world, dtype=np.float64)
        # getWorld2View2_TDOM transposes R, so camera axes must be columns.
        rotation = np.stack((-image_x, image_y, normal), axis=1)
        translation = torch.tensor(reference_camera.T).clone()
        center_x, center_y = window.center_plane
        translation[0] = center_x
        translation[1] = -center_y
        scale = 0.13
        far = 1.0
        fov_x = 2 * math.atan(window.width_world * scale / (2 * far))
        fov_y = 2 * math.atan(window.height_world * scale / (2 * far))
        self.TDOM_Cam = DummyCamera(
            rotation,
            translation,
            fov_x,
            fov_y,
            window.width_pixels,
            window.height_pixels,
        )
        self.TDOM_pipeline = DummyPipeline()
        product_key = "stage_{:06d}_{}".format(
            int(self.imagenum), training_type.lower()
        )
        self.TDOM_alignment = {
            "schema_version": 2,
            "product_key": product_key,
            "geometry_stage": int(self.imagenum),
            "training_type": training_type,
            "grid": grid.to_dict(),
            "window": window.to_dict(),
            "image_width": int(self.TDOM_Cam.image_width),
            "image_height": int(self.TDOM_Cam.image_height),
            "fov_x_radians": float(self.TDOM_Cam.FoVx),
            "fov_y_radians": float(self.TDOM_Cam.FoVy),
            "requested_gsd": {"x": float(self.GSD_x), "y": float(self.GSD_y)},
            "effective_gsd": {"x": grid.gsd[0], "y": grid.gsd[1]},
            "extent": {
                "plane_bounds": list(window.plane_bounds),
                "width_world": window.width_world,
                "height_world": window.height_world,
            },
            "crs": grid.crs,
            "dataset_preflight": (
                {
                    "scope": "run_root",
                    "path": os.path.relpath(
                        self.args.dataset_preflight_path,
                        self.args.Model_Path_Dir,
                    ),
                    "sha256": self.args.dataset_preflight_sha256,
                    "authoritative_exact_complete": (
                        self.args.semantic_authoritative_exact_complete
                    ),
                }
                if self.args.enable_semantics
                else None
            ),
        }

    def _prepare_semantic_map_safely(self, training_type):
        """Keep every semantic map-preparation failure outside geometry."""
        try:
            if self.semantic_map_error is not None:
                raise RuntimeError(self.semantic_map_error)
            self._prepare_tdom_publication(training_type)
        except Exception as error:
            self.TDOM_Cam = None
            self.TDOM_alignment = None
            self.TDOM_pipeline = None
            SemanticWarning(
                "TDOM map preparation failed; RGB geometry continues: {}".format(
                    error
                )
            )

    def _render_legacy_tdom(self):
        """RGB-only runs no longer publish an ambiguous progress artifact."""
        return

    def _training_progress_due(self, training_type, final=False):
        if not self.args.render_training_progress:
            return False
        if final or training_type == "Final_Refinement":
            return True
        return int(self.imagenum) in self.training_progress_stages

    @staticmethod
    def _camera_order_key(camera):
        value = str(camera.image_name)
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    @staticmethod
    def _project_world_points(camera, points):
        points = np.asarray(points, dtype=np.float32)
        points = torch.as_tensor(
            points,
            dtype=torch.float32,
            device=camera.full_proj_transform.device,
        )
        homogeneous = torch.cat(
            (points, torch.ones_like(points[:, :1])), dim=1
        )
        clip = homogeneous @ camera.full_proj_transform
        ndc = clip[:, :2] / clip[:, 3:].clamp_min(1e-8)
        pixels = torch.empty_like(ndc)
        pixels[:, 0] = (
            (ndc[:, 0] + 1.0) * float(camera.image_width) - 1.0
        ) * 0.5
        pixels[:, 1] = (
            (ndc[:, 1] + 1.0) * float(camera.image_height) - 1.0
        ) * 0.5
        return pixels.detach().cpu().tolist()

    def _training_progress_camera(self):
        factor = TRAINING_PROGRESS_MAP_DOWNSAMPLE
        width = max(2, int(self.TDOM_Cam.image_width) // factor)
        height = max(2, int(self.TDOM_Cam.image_height) // factor)
        width += width % 2
        height += height % 2
        return DummyCamera(
            self.TDOM_Cam.R,
            self.TDOM_Cam.T,
            self.TDOM_Cam.FoVx,
            self.TDOM_Cam.FoVy,
            width,
            height,
        )

    def _make_training_progress_capture(self, training_type, final=False):
        if self.TDOM_Cam is None or self.TDOM_alignment is None:
            raise RuntimeError("training progress frame is due but its TDOM camera is missing")
        camera = self._training_progress_camera()
        cameras = sorted(
            self.scene.getTrainCameras(), key=self._camera_order_key
        )
        trajectory_world = [
            item.camera_center.detach().cpu().tolist() for item in cameras
        ]
        trajectory_pixels = self._project_world_points(
            camera, trajectory_world
        )
        grid = self.TDOM_alignment["grid"]
        origin = np.asarray(grid["origin_world"], dtype=np.float32)
        axis_x = np.asarray(grid["pixel_x_axis_world"], dtype=np.float32)
        axis_y = np.asarray(grid["pixel_y_axis_world"], dtype=np.float32)
        basis_pixels = self._project_world_points(
            camera, [origin, origin + axis_x, origin + axis_y]
        )
        column_direction = 1 if basis_pixels[1][0] > basis_pixels[0][0] else -1
        row_direction = 1 if basis_pixels[2][1] > basis_pixels[0][1] else -1
        denominator = max(
            1, int(self.args.EndAtImageNo) - int(self.args.StartFromImageNo)
        )
        metadata = {
            "schema_version": 1,
            "stage": int(self.imagenum),
            "phase": (
                "final_refinement" if final else training_type.lower()
            ),
            "iteration": int(self.AlreadyTrainingIterations),
            "observed_cameras": len(cameras),
            "start_stage": int(self.args.StartFromImageNo),
            "end_stage": int(self.args.EndAtImageNo),
            "progress": min(
                1.0,
                max(
                    0.0,
                    (int(self.imagenum) - int(self.args.StartFromImageNo))
                    / float(denominator),
                ),
            ),
            "window": self.TDOM_alignment["window"],
            "grid": grid,
            "map_downsample": TRAINING_PROGRESS_MAP_DOWNSAMPLE,
            "column_plane_direction": column_direction,
            "row_plane_direction": row_direction,
            "trajectory_pixels": trajectory_pixels,
        }
        # Everything captured here belongs to this training boundary. No live
        # trainer/model/camera references are read by the background callback.
        return ProgressCapture(
            FrozenCameraSnapshot(camera), self.background.detach().clone(),
            self.semantic_tdom_config, copy.deepcopy(metadata), self.DataSaver_queue,
            time.perf_counter())

    def _capture_training_progress_tdom(self, training_type, final=False):
        timing = dict(requested=False, semantic_wait_seconds=0.0,
                      rgb_tdom_seconds=0.0, semantic_tdom_seconds=0.0,
                      frame_enqueue_seconds=0.0, capture_backpressure_seconds=0.0,
                      capture_submit_seconds=0.0, asynchronous=False)
        if not self._training_progress_due(training_type, final=final):
            return timing
        if not final:
            queued = getattr(self, "_queued_progress_capture", None)
            if queued is None or queued[0] != (int(self.imagenum), training_type):
                raise RuntimeError("training progress stage was not queued with semantic synchronization")
            self._queued_progress_capture = None
            return queued[1]
        if self.semantic_hook is None or self.last_semantic_sync_result is None:
            raise RuntimeError("training progress requires a synchronized semantic state")
        started = time.perf_counter()
        self.semantic_hook.wait_for_idle()
        timing['semantic_wait_seconds'] = time.perf_counter() - started
        status = self.semantic_hook.current_status(self.imagenum)
        capture = self._make_training_progress_capture(training_type, final=True)
        measured = capture(self.semantic_hook.core, self.gaussians, status)
        timing.update(requested=True, rgb_tdom_seconds=measured['rgb_tdom_seconds'],
                      semantic_tdom_seconds=measured['semantic_tdom_seconds'])
        return timing

    # 3DGS模型训练核心函数
    def Train_Gaussians(self, iteration, Training_Type):
        # self.Diary.write('\n')

        # A map request may only use a TDOM camera created by this exact stage.
        # The task-level grid remains fixed while per-event camera data resets.
        self.TDOM_Cam = None
        self.TDOM_alignment = None
        self.TDOM_pipeline = None

        Start_From_Its = self.AlreadyTrainingIterations

        # 初始化影像栈以及进度条
        viewpoint_stack = None
        EvaluateRender = False
        progress_bar = tqdm(range(0, Start_From_Its + iteration), desc=f"{Training_Type} Training progress",
                            initial=Start_From_Its)
        progress_reporter = TrainingProgressReporter(
            progress_bar,
            interval=TRAINING_PROGRESS_UPDATE_INTERVAL,
            initial_ema_loss=self.ema_loss_for_log,
        )
        stage_input_cache = StageTrainingInputCache(
            trainable_cameras(self.NewCams) if Training_Type == "On_The_Fly" else (),
            device=self.bg.device,
        )
        retain_visibility = should_retain_visibility(
            Training_Type,
            self.args.No_Sampling,
            reset_pending=self.ResetOpacityTime >= 1,
        )

        sub_iteration = 1
        while sub_iteration <= iteration:
            # 如果需要给前端渲染影像，那么必须等这一张影像的训练至少进行一半才可以进行
            if not self.GaussianTrainer_queue.empty() or self.commond is not None:
                if self.commond is None:
                    self.commond = self.GaussianTrainer_queue.get()
                if self.commond[0] == "Render_New" and sub_iteration > iteration / 2:
                    with torch.no_grad():
                        NewCam = self.commond[1]
                        render_pkg = render(NewCam, self.gaussians, self.args, self.bg, Render_Mask=NewCam.Tri_Mask)
                        image = render_pkg["render"]

                        # 向前端传递信息
                        msg = [image]
                        self.DataPreProccesser_queue.put(msg)

                        self.commond = None
                elif self.commond[0] == "Render_New" and sub_iteration <= iteration / 2:
                    pass
                elif self.commond[0] == "Final_Refinement_Prepare":
                    pass
                elif self.commond[0] == "Final_Refinement":
                    pass
                elif self.commond[0] == "RenderNewDone":
                    self.RenderNewDone = True
                    self.commond = None
                else:
                    print(self.commond)
                    self.commond = None


            # for sub_iteration in range(iteration):
            # 更新模型已训练次数
            self.AlreadyTrainingIterations += 1

            # 当摄影机视点栈堆为空时：
            if not viewpoint_stack:
                if Training_Type != "On_The_Fly" or self.args.No_Adaptive:
                    # 将Scene类中所有的self.train_cameras中指定缩放比例的训练影像及其相关信息存入到摄影机视点栈堆中
                    viewpoint_stack = self.scene.getTrainCameras().copy()

                    # 如果是模型初始化训练，那么需要先将初始模型就已存在的影像的已训练次数置为0
                    if Training_Type == "Initialization" and len(
                            list(self.ImagesAlreadyBeTrainedIterations.keys())) == 0:
                        for _ in range(len(viewpoint_stack)):
                            self.ImagesAlreadyBeTrainedIterations[viewpoint_stack[_].image_name] = int(
                                self.args.Mean * self.args.IterationFirstScene / 30000)
                else:
                    # 在渐进式训练过程中，将根据特殊规则选取用于训练的影像
                    viewpoint_stack = self.GetTraining_Viewpoints_SingleImage(single=self.args.Single)

                viewpoint_stack = trainable_cameras(viewpoint_stack)
                if not viewpoint_stack:
                    raise RuntimeError(
                        "training stage has no camera with a valid sparse support region"
                    )

            # 随机从摄影机视点栈堆中任意取出一个影像以及其视点信息
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            render_mask = stage_input_cache.get_mask(viewpoint_cam)

            # 学习率下降
            if Training_Type == "Initialization" or self.args.No_Adaptive:
                lr = self.gaussians.update_learning_rate(self.AlreadyTrainingIterations)
            else:
                lr = self.gaussians.SingleImage_Update_Lr(self.ImagesAlreadyBeTrainedIterations,
                                                          self.args, viewpoint_cam,
                                                          self.args.IterationPerMergeScene)

            # 将球谐函数的阶数提高一阶，最多至3阶
            if self.AlreadyTrainingIterations % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # 在进行渐进式训练前，保证球谐函数已经提升至最高
            if self.gaussians.active_sh_degree < self.args.sh_degree:
                self.gaussians.oneupSHdegree()

            # 计算损失
            if Training_Type != "On_The_Fly":
                # 进行影像渲染，渲染指定视点的影像
                render_pkg = render(viewpoint_cam, self.gaussians, self.args, self.bg, Render_Mask=render_mask)
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    load_distribution
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter_bool"],
                    render_pkg["radii"],
                    render_pkg["load_distribution"]
                )

                if retain_visibility:
                    self.Image_Visibility[viewpoint_cam.image_name] = visibility_filter

                # 计算损失
                if self.args.Use_Tri_Mask:
                    image = image * render_mask
                    load_distribution = load_distribution[render_mask.squeeze(0)]

                gt_image = stage_input_cache.get(viewpoint_cam)
                Ll1 = l1_loss(image, gt_image)
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                loss = (1.0 - self.args.lambda_dssim) * Ll1 + self.args.lambda_dssim * (1.0 - ssim_value)

            # 目前暂时采用与其他情况相同的训练策略
            else:
                # 进行影像渲染，渲染指定视点的影像
                render_pkg = render(viewpoint_cam, self.gaussians, self.args, self.bg,
                                    Render_Mask=render_mask)
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    load_distribution
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter_bool"],
                    render_pkg["radii"],
                    render_pkg["load_distribution"]
                )

                if retain_visibility:
                    self.Image_Visibility[viewpoint_cam.image_name] = visibility_filter

                # 计算损失
                if self.args.Use_Tri_Mask:
                    image = image * render_mask
                    load_distribution = load_distribution[render_mask.squeeze(0)]

                gt_image = stage_input_cache.get(viewpoint_cam)
                Ll1 = l1_loss(image, gt_image)
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                loss = (1.0 - self.args.lambda_dssim) * Ll1 + self.args.lambda_dssim * (1.0 - ssim_value)


            # 完成这一次训练后，让当前训练影像的已训练次数+1
            if Training_Type != "Initialization":
                self.ImagesAlreadyBeTrainedIterations[viewpoint_cam.image_name] = self.ImagesAlreadyBeTrainedIterations[
                                                                                      viewpoint_cam.image_name] + 1

            # self.GaussianDiary.write(f"Iteration: {self.AlreadyTrainingIterations}, "
            #                          f"Image Name: {viewpoint_cam.image_name}, "
            #                          f"Image Training Times: {self.ImagesAlreadyBeTrainedIterations[viewpoint_cam.image_name]}, "
            #                          f"Lr: {lr}, "
            #                          f"spatial_lr_scale: {self.scene.cameras_extent}, "
            #                          f"Gaussians: {visibility_filter.shape[0]}\n")

            # 反向传播
            loss.backward()

            with torch.no_grad():
                # Loss logging is telemetry only. Sampling it avoids forcing a
                # device-wide synchronization after every optimizer iteration.
                progress_reporter.maybe_update(
                    loss,
                    step=sub_iteration,
                    total_steps=iteration,
                )
                self.ema_loss_for_log = progress_reporter.ema_loss
                if sub_iteration == iteration:
                    EvaluateRender = True

                GaussianPruned = False

                # 判断是否需要重新设置可见度（注意，此时在渐进式训练过程中暂时不进行该步骤）
                # 如果使用了No_Sampling，则也可以使用该步骤
                if self.AlreadyTrainingIterations % self.args.opacity_reset_interval == 0:
                    '''
                    if (Training_Type == "Initialization") or (
                            Training_Type == "Final_Refinement" and
                            self.args.FinalOptimizationIterations - (
                                    self.AlreadyTrainingIterations - Start_From_Its) > self.args.opacity_reset_interval and
                            self.AlreadyTrainingIterations <= Start_From_Its + iteration / 2):'''
                    if (Training_Type == "Initialization" or self.args.No_Sampling):
                        self.ResetOpacityTime += 1

                # 如果处于模型初始化训练阶段
                if (Training_Type == "Initialization" and self.AlreadyTrainingIterations < self.args.densify_until_iter) or self.args.No_Sampling:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # 稠密化
                    if self.AlreadyTrainingIterations > self.args.densify_from_iter and self.AlreadyTrainingIterations % self.args.densification_interval == 0 and self.ResetOpacityTime < 1:
                        size_threshold = 20 if self.AlreadyTrainingIterations > self.args.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.args.densify_grad_threshold, self.args.opacity_threshold,
                                                         self.scene.cameras_extent, size_threshold, radii)
                        budget_pruned = self.gaussians.prune_to_max_count(
                            self.args.GaussiansNumMax,
                            self.args.GaussianBudgetSlack,
                        )
                        if budget_pruned:
                            print(
                                "Hard Gaussian budget pruned {} points; "
                                "retained {}".format(
                                    budget_pruned, self.args.GaussiansNumMax
                                )
                            )
                        GaussianPruned = True
                '''
                elif Training_Type == "Final_Refinement" and self.AlreadyTrainingIterations <= Start_From_Its + iteration / 2:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # 稠密化
                    if self.AlreadyTrainingIterations % self.args.densification_interval == 0 and self.ResetOpacityTime < 1:
                        size_threshold = 20 if self.AlreadyTrainingIterations > self.args.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.args.densify_grad_threshold, self.args.opacity_threshold, self.scene.cameras_extent, size_threshold, radii)
                        GaussianPruned = True'''

                # 重新设置可见度
                if self.ResetOpacityTime >= 1 and (not (self.Drone_Image_Block or self.Progressive_Training_Block)) and self.AlreadyTrainingIterations - self.StartFromTrainingIterations >= 100:
                    print(f"[{self.AlreadyTrainingIterations} its] => Reset Opacity!!")
                    self.gaussians.reset_opacity(self.Image_Visibility)

                    self.Image_Visibility = {}

                    self.ResetOpacityTime -= 1

                if GaussianPruned:
                    self.Image_Visibility = {}

                # 优化器参数更新
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                # if sub_iteration + 1 == iteration:
                #     if Training_Type == "On_The_Fly":
                #         self.Phase_End("GS_Expanding")
                #     elif Training_Type == "Final_Refinement":
                #         self.Phase_End("Final_Refinement")
                #     elif Training_Type == "Initialization":
                #         self.Phase_End("Initial_GS_Optimization")

                # 每训练一定次数输出一次模型评估结果
                self.Evaluate(Training_Type, EvaluateRender)
                EvaluateRender = False

            sub_iteration = sub_iteration + 1

        if iteration > 0:
            # The final iteration's Python locals otherwise retain the render
            # and autograd graph while the hard-budget prune allocates compacted
            # model and optimizer tensors. Release them before that copy.
            del render_pkg, image, viewspace_point_tensor, visibility_filter
            del radii, load_distribution, gt_image, Ll1, ssim_value, loss
            del render_mask
            torch.cuda.empty_cache()

        stage_input_cache.clear()
        del stage_input_cache, progress_reporter

        # 在完成这一张影像的训练之后
        if Training_Type == "On_The_Fly" and not self.args.No_Sampling:
            with torch.no_grad():
                # 进行一次高斯球剔除
                prune_mask = (self.gaussians.get_opacity < self.args.opacity_threshold).squeeze()
                self.gaussians.prune_points(prune_mask)
                print(f"Prune Gaussians Num: {prune_mask.sum()}")
                budget_pruned = self.gaussians.prune_to_max_count(
                    self.args.GaussiansNumMax,
                    self.args.GaussianBudgetSlack,
                )
                if budget_pruned:
                    print(
                        "Hard Gaussian budget pruned {} points; retained {}".format(
                            budget_pruned, self.args.GaussiansNumMax
                        )
                    )

        if self.args.render_TDOM:
            print("GaussianTrainer: Prepare TDOM publication...")
            with torch.no_grad():
                if self.args.enable_semantics:
                    self._prepare_semantic_map_safely(Training_Type)
                else:
                    self._prepare_tdom_publication(Training_Type)
                    self._render_legacy_tdom()

    # 对当前高斯模型进行一次评估
    def Evaluate(self, Training_Type, EvaluateRender):
        if (
            Training_Type == "Final_Refinement"
            and self.Finish
            and getattr(self.args, "skip_final_evaluation", False)
        ):
            write_json_atomic(
                Path(self.args.metrics_dir) / "final_evaluation.json",
                {
                    "schema_version": 1,
                    "status": "skipped",
                    "reason": "--skip_final_evaluation",
                    "iteration": int(self.AlreadyTrainingIterations),
                    "gaussians": int(self.gaussians.get_xyz.shape[0]),
                    "complete_metrics": False,
                },
            )
            print("Final quality evaluation skipped (--skip_final_evaluation).")
            return
        DoEvaluate = False
        if Training_Type == "Initialization" and (
                self.AlreadyTrainingIterations % self.args.InitialTrainingEvaluateInterval == 0 or EvaluateRender):
            DoEvaluate = True
        elif (Training_Type == "On_The_Fly" or Training_Type == "Final_Refinement") and EvaluateRender:
            DoEvaluate = True

            if Training_Type == "Final_Refinement":
                self.ChosenImageNames = []
        elif Training_Type == "Final_Refinement" and self.AlreadyTrainingIterations % self.args.FinalTrainingEvaluateInterval == 0:
            DoEvaluate = True

        if self.args.NoDebug and Training_Type != "Final_Refinement":
            DoEvaluate = False

        if DoEvaluate:
            self.EvaluateModel(l1_loss, render, (self.pp.extract(self.args), self.background), DoEvaluate)

    # 打印模型的评价结果
    def EvaluateModel(self, l1_loss, renderFunc, renderArgs, DoEvaluate):
        # Report test and samples of training set
        torch.cuda.empty_cache()
        if DoEvaluate:
            if self.Finish:
                lpips_vgg = lpips.LPIPS(net='vgg').eval().to("cuda")
                progress_bar = tqdm(range(0, self.imagenum), desc=f"Final Evaluate...", initial=0)

            if self.EvaluateDiary is None:
                self.EvaluateDiary = open(
                    os.path.join(self.args.metrics_dir, "evaluations.jsonl"),
                    "a",
                )

            if self.ChosenImageNames == []:
                validation_configs = ({'name': 'test', 'cameras': self.scene.getTestCameras()},
                                      {'name': 'train', 'cameras': self.scene.getTrainCameras()})
            else:
                validation_configs = ({'name': 'test', 'cameras': self.scene.getTestCameras()},
                                      {'name': 'train', 'cameras': [traincam for traincam in self.scene.getTrainCameras() if
                                                                    traincam.image_name in self.ChosenImageNames]})

            camera_rows = []
            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    lpips_test = 0.0
                    ssim_test = 0.0

                    PSNRs = []
                    ErrorImageNum = 0
                    for idx, viewpoint in enumerate(config['cameras']):
                        evaluation_mask = viewpoint.Tri_Mask
                        if evaluation_mask.shape[0] == 2:
                            evaluation_mask = torch.ones(
                                1,
                                dtype=torch.bool,
                                device=evaluation_mask.device,
                            )
                        render_pkg = renderFunc(
                            viewpoint,
                            self.gaussians,
                            *renderArgs,
                            Render_Mask=evaluation_mask,
                        )
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        gt_image = torch.clamp(
                            viewpoint.original_image.to("cuda"), 0.0, 1.0
                        )
                        l1_test += l1_loss(image, gt_image).mean().double()

                        PSNR = psnr(image, gt_image, evaluation_mask).mean().double()
                        psnr_test += PSNR
                        PSNRs.append(PSNR)

                        if self.Finish:
                            lpips_test += Get_lpips(image, gt_image, lpips_vgg)['vgg']
                            ssim_test += ssim_chw(gt_image, image)

                            if (
                                config["name"] == "train"
                                and getattr(self.args, "render_all_cameras", False)
                            ):
                                image_id = Path(str(viewpoint.image_name)).name
                                output_dir = Path(self.args.renders_dir) / "cameras"
                                output_dir.mkdir(parents=True, exist_ok=True)
                                Image.fromarray(tdom_to_uint8(image)).save(
                                    output_dir / (image_id + ".jpg")
                                )
                                visibility = render_pkg["visibility_filter"]
                                camera_rows.append(
                                    {
                                        "camera_id": image_id,
                                        "iteration": int(
                                            self.AlreadyTrainingIterations
                                        ),
                                        "psnr": float(PSNR),
                                        "visible_gaussians": int(
                                            torch.count_nonzero(visibility).item()
                                        ),
                                        "render": (
                                            "renders/cameras/{}.jpg".format(image_id)
                                        ),
                                    }
                                )

                            progress_bar.update(1)

                        del render_pkg, image, gt_image

                    if self.Finish:
                        progress_bar.close()

                    psnr_test /= len(config['cameras']) - ErrorImageNum
                    ssim_test /= len(config['cameras']) - ErrorImageNum
                    lpips_test /= len(config['cameras']) - ErrorImageNum
                    l1_test /= len(config['cameras']) - ErrorImageNum
                    print("\n[ITER {}] Evaluating {}: L1 {} Gaussians {} PSNR {} SSIM {} LPIPS {}".format(self.AlreadyTrainingIterations,
                                                                                         config['name'], l1_test,
                                                                                         self.gaussians.get_xyz.shape[0],
                                                                                         psnr_test, ssim_test, lpips_test))

                    evaluation_record = {
                        "iteration": int(self.AlreadyTrainingIterations),
                        "split": config["name"],
                        "l1": float(l1_test),
                        "gaussians": int(self.gaussians.get_xyz.shape[0]),
                        "psnr": float(psnr_test),
                        "ssim": float(ssim_test),
                        "lpips": float(lpips_test),
                        "complete_metrics": bool(self.Finish),
                    }
                    self.EvaluateDiary.write(
                        json.dumps(evaluation_record, sort_keys=True) + "\n"
                    )
                    self.EvaluateDiary.flush()
                    if self.Finish and config["name"] == "train":
                        write_json_atomic(
                            Path(self.args.metrics_dir) / "final_evaluation.json",
                            {"schema_version": 1, **evaluation_record},
                        )

            if camera_rows:
                self._write_per_camera_metrics(camera_rows)

                # self.Diary.write(
                #     "[ITER {}] Evaluating {}: L1 {} Gaussians {} PSNR {}\n".format(self.AlreadyTrainingIterations,
                #                                                                    config['name'], l1_test,
                #                                                                    self.gaussians.get_xyz.shape[0],
                #                                                                    psnr_test))
                #
                # if self.EvaluateDiary is not None and len(PSNRs) != 0:
                #     self.EvaluateDiary.write("-----------------------------------------------------------\n")
                #     self.EvaluateDiary.write(f"Iterations: {self.AlreadyTrainingIterations}\n")
                #     for i in range(len(PSNRs)):
                #         self.EvaluateDiary.write(f"{config['cameras'][i].image_name}: {PSNRs[i]}\n")
        else:
            print("\n[ITER {}] L1 {} Gaussians {} PSNR {} SSIM {} LPIPS {}".format(
                0.0,
                0.0,
                self.gaussians.get_xyz.shape[0],
                0.0, 0.0, 0.0))

    # 仅训练当前新传入的影像
    def GetTraining_Viewpoints_SingleImage(self, single=False):
        viewpoint_stack = []
        new_cameras = trainable_cameras(self.NewCams)
        if single:
            for _ in range(self.args.IterationPerMergeScene):
                if new_cameras:
                    viewpoint_stack.append(new_cameras[0])
        else:
            for _ in range(self.args.IterationPerMergeScene):
                viewpoint_stack.extend(new_cameras)

        all_viewpoints = trainable_cameras(self.scene.getTrainCameras())
        remaining = self.CurrentGlobalOptimizationIterations
        if remaining > 0 and not all_viewpoints:
            raise RuntimeError(
                "global optimization has no camera with a valid sparse support region"
            )
        while remaining > 0:
            take = min(remaining, len(all_viewpoints))
            viewpoint_stack.extend(random.sample(all_viewpoints, take))
            remaining -= take

        return viewpoint_stack

    # 为所有影像进行一次渲染并输出一些评估结果
    def Render_Evaluate_All_Images(self):
        with torch.no_grad():
            ALL_viewpoint_stack = self.scene.getTrainCameras().copy()
            progress_bar = tqdm(range(len(ALL_viewpoint_stack)), desc="Rendering progress")
            output_dir = Path(self.args.renders_dir) / "cameras"
            output_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for i in range(len(ALL_viewpoint_stack)):
                progress_bar.update(1)

                viewpoint_cam = ALL_viewpoint_stack[i]
                render_pkg = render(viewpoint_cam, self.gaussians, self.args, self.bg,
                                    Render_Mask=viewpoint_cam.Tri_Mask)
                image, visibility_filter = render_pkg["render"], render_pkg["visibility_filter"]

                ImageOutput = Image.fromarray(tdom_to_uint8(image))
                image_id = Path(str(viewpoint_cam.image_name)).name
                ImageOutput.save(output_dir / (image_id + ".jpg"))

                gt_image = viewpoint_cam.original_image.cuda()
                PSNR = psnr(image, gt_image, viewpoint_cam.Tri_Mask)
                PSNR = float(PSNR.mean().item())
                rows.append(
                    {
                        "camera_id": image_id,
                        "iteration": int(self.AlreadyTrainingIterations),
                        "psnr": float(PSNR),
                        "visible_gaussians": int(
                            torch.count_nonzero(visibility_filter).item()
                        ),
                        "render": "renders/cameras/{}.jpg".format(image_id),
                    }
                )
                del render_pkg, image, visibility_filter, gt_image
            progress_bar.close()
            self._write_per_camera_metrics(rows)

    def _write_per_camera_metrics(self, rows):
        metrics_path = Path(self.args.metrics_dir) / "per_camera.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "camera_id",
                    "iteration",
                    "psnr",
                    "visible_gaussians",
                    "render",
                ),
            )
            writer.writeheader()
            writer.writerows(rows)

    # 保存当前模型
    def _checkpoint_payload(self, phase):
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "stage": int(self.imagenum),
            "gaussians": self.gaussians.capture(),
            "iteration": int(self.AlreadyTrainingIterations),
            "cameras": [
                camera_to_state(camera)
                for camera in self.scene.getTrainCameras()
            ],
            "cameras_extent": float(self.scene.cameras_extent),
            "images_already_trained_iterations": dict(
                self.ImagesAlreadyBeTrainedIterations
            ),
            "progressive_transition_index": int(
                self.progressive_transition_index
            ),
            "reset_opacity_time": int(self.ResetOpacityTime),
            "rng_state": capture_rng_state(),
            "data_preprocessor_rng_state": self.data_preprocessor_rng_state,
            "dataset_preflight_sha256": getattr(
                self.args, "dataset_preflight_sha256", None
            ),
            "source_path_dir": str(Path(self.args.Source_Path_Dir).resolve()),
            "phase": phase,
        }

    def _save_training_checkpoint(self, path, phase="stage_complete"):
        save_training_checkpoint(path, self._checkpoint_payload(phase))

    def _save_recovery_checkpoint(self):
        checkpoint_path = Path(self.args.work_dir) / "recovery" / "latest.pt"
        print(
            "\n[Recovery At Stage {}] Saving {}".format(
                self.imagenum, checkpoint_path
            )
        )
        self._save_training_checkpoint(checkpoint_path, phase="stage_complete")

    def _write_final_camera_inventory(self):
        """Publish the cumulative camera state used by the final model.

        A progressive source stage may contain only a local overlap snapshot.
        The trainer, however, keeps the persistent union and applies later pose
        replacements in memory.  Publishing the final stage's source
        ``cameras.json`` would therefore describe only that local snapshot,
        not the cameras that actually trained and evaluated the model.
        """
        cameras = sorted(
            self.scene.getTrainCameras(), key=self._camera_order_key
        )
        identities = [str(camera.image_name) for camera in cameras]
        if len(set(identities)) != len(identities):
            raise RuntimeError(
                "final cumulative camera inventory contains duplicate identities"
            )
        records = [
            camera_to_JSON(index, camera)
            for index, camera in enumerate(cameras)
        ]
        write_json_atomic(
            Path(self.scene.model_path) / "cameras.json", records
        )

    def GS_Save(self, UseSecondPath=True, final=False):
        if not final:
            raise ValueError("GS_Save is reserved for the final published model")
        imagedirname = self.imagenum
        if self.args.save_final_training_state:
            print(
                "\n[Save At Image {}] Saving Final Training State".format(
                    imagedirname
                )
            )
            checkpoint_path = os.path.join(
                self.args.stage_output_dir,
                str(imagedirname),
                f"chkpnt{imagedirname}.pth",
            )
            self._save_training_checkpoint(
                checkpoint_path,
                phase="run_complete",
            )

        print("\n[Save At Image {}] Saving Gaussians".format(imagedirname))
        self.scene.save(imagedirname)
        self._write_final_camera_inventory()

    # 该函数将被送入到一个线程中，用于渐进式训练中的数据读取以及预处理
    def run(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        # Configure this process before starting the semantic executor. Keep
        # OpenCV preprocessing within the existing Torch/OMP CPU budget rather
        # than changing its global thread policy inside concurrent projections.
        cv2.setNumThreads(max(1, torch.get_num_threads()))
        torch.cuda.set_device(
            torch.device("cuda:{}".format(int(self.args.gpu)))
        )
        while True:
            Commond_from_DataPreProccesser = self.GaussianTrainer_queue.get()
            if Commond_from_DataPreProccesser is not None:
                if Commond_from_DataPreProccesser[0] == "init":
                    self.InitialGaussianTrain()
                elif Commond_from_DataPreProccesser[0] == "stop":
                    break
                elif Commond_from_DataPreProccesser[0] == "Render_New":
                    with torch.no_grad():
                        NewCam = Commond_from_DataPreProccesser[1]
                        render_pkg = render(NewCam, self.gaussians, self.args, self.bg, Render_Mask=NewCam.Tri_Mask)
                        image = render_pkg["render"]

                        # 向前端传递信息
                        msg = [image]
                        self.DataPreProccesser_queue.put(msg)
                elif Commond_from_DataPreProccesser[0] == "Progressive_Train":
                    with torch.no_grad():
                        self.TDOM_Cam = None
                        self.TDOM_alignment = None
                        stage_started = time.perf_counter()
                        torch.cuda.reset_peak_memory_stats()
                        print("GaussianTrainer: Get Information from DataPreProccesser...")

                        self.scene.scene_info_traincam = Commond_from_DataPreProccesser[1]
                        self.NewCams = Commond_from_DataPreProccesser[2]
                        UpdatedTrainCamsPosList = Commond_from_DataPreProccesser[3]
                        self.scene.cameras_extent = Commond_from_DataPreProccesser[6]
                        progressive_pcd = Commond_from_DataPreProccesser[4]
                        if not self.args.No_Sampling and progressive_pcd is not None:
                            self.gaussians.expand_from_pcd(progressive_pcd, self.scene.cameras_extent)
                        if progressive_pcd is not None:
                            self.scene.basic_pcd = progressive_pcd
                        self.scene.model_path = Commond_from_DataPreProccesser[5]
                        NewImagesNum = Commond_from_DataPreProccesser[7]
                        self.data_preprocessor_rng_state = Commond_from_DataPreProccesser[8]

                        self.imagenum = progressive_stage_number(
                            Commond_from_DataPreProccesser[5]
                        )
                        print(f"-----------------------------This is {self.imagenum} image-----------------------------")

                        # 还原为 dict
                        worlds = UpdatedTrainCamsPosList[0]
                        projs = UpdatedTrainCamsPosList[1]
                        fulls = UpdatedTrainCamsPosList[2]
                        centers = UpdatedTrainCamsPosList[3]
                        names = UpdatedTrainCamsPosList[4]
                        camera_infos = UpdatedTrainCamsPosList[5]
                        replacements = UpdatedTrainCamsPosList[6]
                        UpdatedTrainCamsPos = {}
                        for i, name in enumerate(names):
                            UpdatedTrainCamsPos[name] = [
                                worlds[i],
                                projs[i],
                                fulls[i],
                                centers[i]
                            ]

                        # 更新位姿
                        for i in range(len(self.scene.train_cameras[1.0])):
                            image_name = self.scene.train_cameras[1.0][i].image_name
                            if image_name in UpdatedTrainCamsPos:
                                update_index = names.index(image_name)
                                replacement = replacements[update_index]
                                if replacement is not None:
                                    self.scene.train_cameras[1.0][i] = replacement
                                    continue
                                info = camera_infos[update_index]
                                camera = self.scene.train_cameras[1.0][i]
                                camera.R = info.R
                                camera.T = info.T
                                camera.FoVx = info.FovX
                                camera.FoVy = info.FovY
                                camera.calibration = info.calibration
                                camera.world_view_transform = UpdatedTrainCamsPos[image_name][0].cuda()
                                camera.projection_matrix = UpdatedTrainCamsPos[image_name][1].cuda()
                                camera.full_proj_transform = UpdatedTrainCamsPos[image_name][2].cuda()
                                camera.camera_center = UpdatedTrainCamsPos[image_name][3].cuda()

                        # 将新的影像加入
                        for i in range(len(self.NewCams)):
                            self.scene.train_cameras[1.0].append(Commond_from_DataPreProccesser[2][i])

                        self.ImagesAlreadyBeTrainedIterations_Set()

                        # 向前端传递信息，表示开始读取后续的影像
                        self.SendImagePos_to_DataPreProccesser()

                    # 对扩张后的场景进行训练
                    if not self.args.skip_MergeSceneTraining:
                        self.progressive_transition_index += 1
                        local_iterations, global_iterations = progressive_stage_iterations(
                            new_image_count=NewImagesNum,
                            transition_index=self.progressive_transition_index,
                            local_iterations_per_image=self.args.IterationPerMergeScene,
                            global_interval=self.args.GlobalOptimizationInterval,
                            global_iterations=self.args.GlobalOptimizationIteration,
                        )
                        self.CurrentGlobalOptimizationIterations = global_iterations
                        self.Train_Gaussians(
                            local_iterations + global_iterations,
                            "On_The_Fly",
                        )

                    geometry_seconds = time.perf_counter() - stage_started
                    self.TimeCost["ProgressiveTrain"] += geometry_seconds

                    # WP3 synchronization point: one progressive RGB batch ended.
                    map_prepare_started = time.perf_counter()
                    if self.args.render_TDOM and self.TDOM_Cam is None:
                        with torch.no_grad():
                            self._prepare_semantic_map_safely("On_The_Fly")
                            if not self.args.enable_semantics:
                                self._render_legacy_tdom()
                    map_prepare_seconds = (
                        time.perf_counter() - map_prepare_started
                    )
                    semantic_enqueue_started = time.perf_counter()
                    self._run_semantic_hook("progressive")
                    semantic_enqueue_seconds = (
                        time.perf_counter() - semantic_enqueue_started
                    )
                    with torch.no_grad():
                        video_timing = self._capture_training_progress_tdom(
                            "On_The_Fly"
                        )

                    checkpoint_started = time.perf_counter()
                    if (
                        self.args.recovery_checkpoint_every_stages > 0
                        and self.imagenum
                        % self.args.recovery_checkpoint_every_stages
                        == 0
                    ):
                        with torch.no_grad():
                            self._save_recovery_checkpoint()
                    checkpoint_seconds = time.perf_counter() - checkpoint_started

                    # 等待self.RenderNewDone变为True
                    render_wait_started = time.perf_counter()
                    while not self.RenderNewDone:
                        if self.commond is None:
                            self.commond = self.GaussianTrainer_queue.get()
                        if self.commond[0] == "Render_New":
                            with torch.no_grad():
                                NewCam = self.commond[1]
                                render_pkg = render(NewCam, self.gaussians, self.args, self.bg, Render_Mask=NewCam.Tri_Mask)
                                image = render_pkg["render"]

                                # 向前端传递信息
                                msg = [image]
                                self.DataPreProccesser_queue.put(msg)

                                self.commond = None
                        elif self.commond[0] == "Final_Refinement_Prepare":
                            if self.args.recovery_checkpoint_every_stages > 0:
                                with torch.no_grad():
                                    self._save_recovery_checkpoint()
                            self.RenderNewDone = True
                            self.commond = None
                        elif self.commond[0] == "RenderNewDone":
                            self.RenderNewDone = True
                            self.commond = None
                    render_new_wait_seconds = (
                        time.perf_counter() - render_wait_started
                    )

                    # 向前端传递信息，表示这一张影像已经完成训练，可以将下一张影像的相关信息传输到后端
                    online_seconds = time.perf_counter() - stage_started
                    self.TimeCost["ProgressiveEndToEnd"] += online_seconds
                    self.ProgressiveStageTimings.append(
                        {
                            "stage": int(self.imagenum),
                            "new_image_count": int(NewImagesNum),
                            "geometry_seconds": float(geometry_seconds),
                            "online_seconds": float(online_seconds),
                            "map_prepare_seconds": float(
                                map_prepare_seconds
                            ),
                            "semantic_enqueue_seconds": float(
                                semantic_enqueue_seconds
                            ),
                            "checkpoint_seconds": float(checkpoint_seconds),
                            "render_new_wait_seconds": float(
                                render_new_wait_seconds
                            ),
                            "video": video_timing,
                            "gaussian_count": int(
                                self.gaussians.get_xyz.shape[0]
                            ),
                            "camera_count": len(
                                self.scene.getTrainCameras()
                            ),
                            "cuda_allocated_bytes": int(
                                torch.cuda.memory_allocated()
                            ),
                            "cuda_reserved_bytes": int(
                                torch.cuda.memory_reserved()
                            ),
                            "cuda_peak_allocated_bytes": int(
                                torch.cuda.max_memory_allocated()
                            ),
                            "cuda_peak_reserved_bytes": int(
                                torch.cuda.max_memory_reserved()
                            ),
                        }
                    )

                    msg = ["Progressive_finish"]
                    self.DataPreProccesser_queue.put(msg)
                    self.RenderNewDone = False

                elif Commond_from_DataPreProccesser[0] == "Final_Refinement":
                    time1 = time.time()
                    self.TDOM_Cam = None
                    self.TDOM_alignment = None
                    self.scene.model_path = Commond_from_DataPreProccesser[1]
                    if self.args.enable_semantics and self.semantic_hook is None:
                        self._initialize_semantic_hook(force_fresh=True)
                    if self.semantic_hook is not None:
                        try:
                            self.semantic_hook.begin_final()
                        except Exception as error:
                            SemanticWarning(
                                "semantic final-boundary hook failed; RGB geometry continues: {}".format(
                                    error
                                )
                            )
                    if not self.args.skip_GlobalOptimization:
                        self.Train_Gaussians(self.args.FinalOptimizationIterations, "Final_Refinement")

                    time2 = time.time()
                    self.TimeCost["FinalRefinement"] += time2 - time1

                    # Persist the completed RGB model before fallible derived
                    # products (semantic rebuild, map/video capture, evaluation).
                    with torch.no_grad():
                        self.GS_Save(True, final=True)

                    # WP3 closes the online group here; WP5 will rebuild exactly.
                    if self.args.render_TDOM and self.TDOM_Cam is None:
                        with torch.no_grad():
                            self._prepare_semantic_map_safely(
                                "Final_Refinement"
                            )
                            if not self.args.enable_semantics:
                                self._render_legacy_tdom()
                    self._run_semantic_hook("final_refinement", final=True)
                    with torch.no_grad():
                        self._capture_training_progress_tdom(
                            "Final_Refinement", final=True
                        )

                    self.Finish = True

                    msg = [
                        "SaveTimeCost",
                        self.TimeCost,
                        self.imagenum,
                        self.ProgressiveStageTimings,
                    ]
                    self.DataSaver_queue.put(msg)

                    with torch.no_grad():
                        self.Evaluate("Final_Refinement", True)

                    # 像后端传递信息
                    msg = ["GetDemo"]
                    self.DataSaver_queue.put(msg)

                    # 向前端传递信息
                    msg = ["stop"]
                    self.DataPreProccesser_queue.put(msg)

                    self.EvaluateDiary.close()

# 后端，第三个线程，主要用于在不影响训练线程的情况下进行一些输出或者保存的工作
class DataSaver(mp.Process):
    def __init__(self):
        super().__init__()
        self.DataSaver_queue = None
        self.SaveDirRootPath = None
        self.MetricsRootPath = None
        self.VideoRootPath = None
        self.RenderTrainingProgress = False
        self.TrainingProgressFPS = 4.0
        self.LegendEntries = []
        self.TrainingProgressFramePath = None
        self.TrainingProgressFrameNumber = 0

    def MakeDirs(self):
        self.TrainingProgressFramePath = os.path.join(
            self.SaveDirRootPath, "training_progress"
        )
        if self.RenderTrainingProgress:
            os.makedirs(self.TrainingProgressFramePath, exist_ok=True)

    @staticmethod
    def _as_uint8(image):
        if isinstance(image, torch.Tensor):
            return tdom_to_uint8(image)
        value = np.asarray(image)
        if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
            raise ValueError("progress frame must be uint8 [H, W, 3]")
        return value

    def SaveTrainingProgress(self, rgb, semantic, metadata):
        rgb = self._as_uint8(rgb)
        semantic = self._as_uint8(semantic)
        if rgb.shape != semantic.shape:
            raise ValueError("RGB and semantic TDOM frames must be aligned")
        number = self.TrainingProgressFrameNumber
        root = Path(self.TrainingProgressFramePath)
        stem = "{:06d}".format(number)
        Image.fromarray(rgb).save(root / (stem + "_rgb.jpg"), quality=92)
        Image.fromarray(semantic).save(root / (stem + "_semantic.png"))
        payload = dict(metadata)
        payload["frame_index"] = number
        payload["rgb_frame"] = stem + "_rgb.jpg"
        payload["semantic_frame"] = stem + "_semantic.png"
        write_json_atomic(root / (stem + ".json"), payload)
        self.TrainingProgressFrameNumber += 1

    @staticmethod
    def _nice_scale_length(value):
        exponent = math.floor(math.log10(max(value, 1e-12)))
        fraction = value / (10 ** exponent)
        if fraction < 1.5:
            nice = 1.0
        elif fraction < 3.5:
            nice = 2.0
        elif fraction < 7.5:
            nice = 5.0
        else:
            nice = 10.0
        return nice * (10 ** exponent)

    @staticmethod
    def _load_progress_records(frame_directory):
        records = []
        for path in sorted(Path(frame_directory).glob("[0-9]*.json")):
            with path.open("r", encoding="utf-8") as stream:
                record = json.load(stream)
            record["_root"] = str(path.parent)
            records.append(record)
        if not records:
            return []
        expected = list(range(len(records)))
        actual = [int(record["frame_index"]) for record in records]
        if actual != expected:
            raise RuntimeError("training progress frame sequence is not contiguous")
        return records

    @staticmethod
    def _tile_indices(window):
        value = window.get("tile_indices", window)
        return (
            int(value.get("min_x", value.get("tile_min_x"))),
            int(value.get("max_x", value.get("tile_max_x"))),
            int(value.get("min_y", value.get("tile_min_y"))),
            int(value.get("max_y", value.get("tile_max_y"))),
        )

    def _draw_legend(self, canvas, panel_x, map_top, map_height):
        if not self.LegendEntries:
            return
        width = 310
        box_x = panel_x + 12
        box_top = int(map_top)
        box_bottom = min(canvas.shape[0] - 23, box_top + int(map_height))
        if box_bottom <= box_top:
            return
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (box_x, box_top),
            (box_x + width, box_bottom),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        cv2.rectangle(
            canvas,
            (box_x, box_top),
            (box_x + width, box_bottom),
            (72, 72, 72),
            1,
        )
        cv2.putText(
            canvas,
            "Land-cover classes",
            (box_x + 16, box_top + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        content_top = box_top + 62
        content_bottom = box_bottom - 28
        count = len(self.LegendEntries)
        if count == 1:
            row_step = 0
            first_y = (content_top + content_bottom) // 2
        else:
            row_step = min(
                72,
                max(36, (content_bottom - content_top) // (count - 1)),
            )
            used_height = row_step * (count - 1)
            first_y = content_top + (
                content_bottom - content_top - used_height
            ) // 2
        for index, item in enumerate(self.LegendEntries):
            y = first_y + index * row_step
            color = tuple(reversed(tuple(int(v) for v in item["color"])))
            cv2.rectangle(
                canvas,
                (box_x + 16, y - 12),
                (box_x + 40, y + 12),
                color,
                -1,
            )
            cv2.putText(
                canvas,
                "{}  {}".format(item["id"], item["name"]),
                (box_x + 52, y + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

    def _training_progress_layout(self, record, global_window):
        """Fit two aligned maps and an optional legend into a compact canvas."""
        margin = 24
        gutter = 24
        header_height = 78
        footer_height = 22
        legend_width = 334 if self.LegendEntries else 0
        map_height = TRAINING_PROGRESS_HEIGHT - header_height - footer_height

        grid = record["grid"]
        factor = int(record["map_downsample"])
        tile_width = int(grid["tile_shape"][0]) // factor
        tile_height = int(grid["tile_shape"][1]) // factor
        global_min_x, global_max_x, global_min_y, global_max_y = global_window
        global_width = (global_max_x - global_min_x + 1) * tile_width
        global_height = (global_max_y - global_min_y + 1) * tile_height
        if global_width <= 0 or global_height <= 0:
            raise ValueError("training progress global map extent must be positive")

        max_map_width = (
            TRAINING_PROGRESS_WIDTH
            - 2 * margin
            - gutter
            - legend_width
        ) // 2
        scale = min(max_map_width / global_width, map_height / global_height)
        global_display_width = max(1, int(round(global_width * scale)))
        global_display_height = max(1, int(round(global_height * scale)))

        content_width = (
            2 * global_display_width + gutter + legend_width
        )
        width = max(TRAINING_PROGRESS_MIN_WIDTH, content_width + 2 * margin)
        width = min(TRAINING_PROGRESS_WIDTH, width)
        if width % 2:
            width += 1
        content_x = (width - content_width) // 2
        rgb_global_x = content_x
        semantic_global_x = rgb_global_x + global_display_width + gutter
        global_y = header_height + (map_height - global_display_height) // 2
        return {
            "width": width,
            "height": TRAINING_PROGRESS_HEIGHT,
            "scale": scale,
            "header_height": header_height,
            "map_height": map_height,
            "rgb_global_x": rgb_global_x,
            "semantic_global_x": semantic_global_x,
            "legend_x": semantic_global_x + global_display_width,
            "global_y": global_y,
            "global_display_width": global_display_width,
            "global_display_height": global_display_height,
            "tile_width": tile_width,
            "tile_height": tile_height,
        }

    def _compose_training_progress_frame(
        self, record, global_window, total_frames, layout
    ):
        width = int(layout["width"])
        height = int(layout["height"])
        header_height = int(layout["header_height"])
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        root = Path(record["_root"])
        rgb = cv2.imread(str(root / record["rgb_frame"]), cv2.IMREAD_COLOR)
        semantic = cv2.imread(
            str(root / record["semantic_frame"]), cv2.IMREAD_COLOR
        )
        if rgb is None or semantic is None or rgb.shape != semantic.shape:
            raise RuntimeError(
                "training progress source frames are missing or misaligned"
            )

        window = record["window"]
        grid = record["grid"]
        factor = int(record["map_downsample"])
        tile_width = int(layout["tile_width"])
        tile_height = int(layout["tile_height"])
        global_min_x, global_max_x, global_min_y, global_max_y = global_window
        window_min_x, window_max_x, window_min_y, window_max_y = (
            self._tile_indices(window)
        )
        scale = float(layout["scale"])
        display_width = max(1, int(round(rgb.shape[1] * scale)))
        display_height = max(1, int(round(rgb.shape[0] * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
        rgb = cv2.resize(rgb, (display_width, display_height), interpolation=interpolation)
        semantic = cv2.resize(
            semantic,
            (display_width, display_height),
            interpolation=cv2.INTER_NEAREST,
        )
        base_x = int(layout["rgb_global_x"])
        base_y = int(layout["global_y"])
        if int(record["column_plane_direction"]) > 0:
            offset_x = (window_min_x - global_min_x) * tile_width
        else:
            offset_x = (global_max_x - window_max_x) * tile_width
        if int(record["row_plane_direction"]) > 0:
            offset_y = (window_min_y - global_min_y) * tile_height
        else:
            offset_y = (global_max_y - window_max_y) * tile_height
        offset_x = int(round(offset_x * scale))
        offset_y = int(round(offset_y * scale))
        rgb_x, rgb_y = base_x + offset_x, base_y + offset_y
        sem_x = int(layout["semantic_global_x"]) + offset_x
        canvas[rgb_y:rgb_y + display_height, rgb_x:rgb_x + display_width] = rgb
        canvas[rgb_y:rgb_y + display_height, sem_x:sem_x + display_width] = semantic
        for x in (rgb_x, sem_x):
            cv2.rectangle(
                canvas,
                (x, rgb_y),
                (x + display_width - 1, rgb_y + display_height - 1),
                (210, 210, 210),
                1,
            )

        trajectory = record.get("trajectory_pixels", [])
        points = []
        for point in trajectory:
            px = rgb_x + int(round(float(point[0]) * scale))
            py = rgb_y + int(round(float(point[1]) * scale))
            if (
                rgb_x <= px < rgb_x + display_width
                and rgb_y <= py < rgb_y + display_height
            ):
                points.append((px, py))
        if len(points) > 1:
            cv2.polylines(
                canvas,
                [np.asarray(points, dtype=np.int32)],
                False,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if points:
            cv2.circle(canvas, points[-1], 6, (0, 165, 255), -1, cv2.LINE_AA)

        gsd_x = float(grid["gsd"][0]) * factor
        pixels_per_unit = scale / gsd_x
        scale_units = self._nice_scale_length(160.0 / pixels_per_unit)
        scale_pixels = max(1, int(round(scale_units * pixels_per_unit)))
        bar_x = int(layout["rgb_global_x"]) + 24
        bar_y = height - 38
        cv2.line(canvas, (bar_x, bar_y), (bar_x + scale_pixels, bar_y), (255, 255, 255), 3)
        cv2.putText(
            canvas,
            "{:g} map units".format(scale_units),
            (bar_x, bar_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        arrow_x = (
            int(layout["rgb_global_x"])
            + int(layout["global_display_width"])
            - 46
        )
        arrow_y = height - 38
        arrow_delta = 30 if int(record["row_plane_direction"]) > 0 else -30
        cv2.arrowedLine(
            canvas,
            (arrow_x, arrow_y),
            (arrow_x, arrow_y + arrow_delta),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.28,
        )
        cv2.putText(
            canvas,
            "+Y",
            (arrow_x + 8, arrow_y + arrow_delta),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        phase_label = str(record["phase"]).replace("_", " ")
        title = (
            "Stage {stage}/{end_stage}  |  Cameras {observed_cameras}  |  "
            "Iteration {iteration:,}  |  {progress:.1%}  |  {phase_label}  |  "
            "Semantic {semantic_status}  |  Frame {frame}/{total}"
        ).format(
            frame=int(record["frame_index"]) + 1,
            total=total_frames,
            phase_label=phase_label,
            **record
        )
        cv2.putText(
            canvas,
            title,
            (24, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "RGB TDOM + flight trajectory",
            (int(layout["rgb_global_x"]), 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Semantic TDOM",
            (int(layout["semantic_global_x"]), 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        self._draw_legend(
            canvas,
            int(layout["legend_x"]),
            int(layout["global_y"]),
            int(layout["global_display_height"]),
        )
        progress_width = int(round(width * float(record["progress"])))
        cv2.rectangle(canvas, (0, height - 6), (progress_width, height - 1), (0, 180, 255), -1)
        return canvas

    def _encode_training_progress(self):
        records = self._load_progress_records(self.TrainingProgressFramePath)
        if not records:
            return
        grid_ids = {record["grid"]["grid_id"] for record in records}
        directions = {
            (
                int(record["column_plane_direction"]),
                int(record["row_plane_direction"]),
                int(record["map_downsample"]),
            )
            for record in records
        }
        if len(grid_ids) != 1 or len(directions) != 1:
            raise RuntimeError("training progress frames do not share one map grid")
        windows = [record["window"] for record in records]
        tile_windows = [self._tile_indices(item) for item in windows]
        global_window = (
            min(item[0] for item in tile_windows),
            max(item[1] for item in tile_windows),
            min(item[2] for item in tile_windows),
            max(item[3] for item in tile_windows),
        )
        layout = self._training_progress_layout(records[-1], global_window)
        video_width = int(layout["width"])
        video_height = int(layout["height"])
        output_root = Path(self.VideoRootPath)
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / "training_progress.mp4"
        temporary = output_path.with_suffix(".publishing.mp4")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "{}x{}".format(video_width, video_height),
            "-r", str(float(self.TrainingProgressFPS)), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for record in records:
                frame = self._compose_training_progress_frame(
                    record, global_window, len(records), layout
                )
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            process.stderr.close()
            returncode = process.wait()
        except Exception:
            process.kill()
            process.wait()
            if temporary.exists():
                temporary.unlink()
            raise
        if returncode != 0:
            if temporary.exists():
                temporary.unlink()
            raise RuntimeError("ffmpeg failed to encode progress video: " + stderr)
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name,width,height,nb_frames",
                    "-of", "json", str(temporary),
                ],
                text=True,
            )
        )["streams"][0]
        if (
            probe.get("codec_name") != "h264"
            or int(probe.get("width", 0)) != video_width
            or int(probe.get("height", 0)) != video_height
            or int(probe.get("nb_frames", 0)) != len(records)
        ):
            temporary.unlink(missing_ok=True)
            raise RuntimeError("encoded training progress video failed validation")
        os.replace(temporary, output_path)
        public_records = []
        for record in records:
            public = {
                key: value
                for key, value in record.items()
                if not key.startswith("_")
                and key not in {"trajectory_pixels", "rgb_frame", "semantic_frame"}
            }
            public_records.append(public)
        write_json_atomic(
            Path(self.MetricsRootPath) / "training_progress.json",
            {
                "schema_version": 1,
                "video": "videos/training_progress.mp4",
                "codec": "h264",
                "width": video_width,
                "height": video_height,
                "fps": float(self.TrainingProgressFPS),
                "frame_count": len(records),
                "global_tile_window": {
                    "tile_min_x": global_window[0],
                    "tile_max_x": global_window[1],
                    "tile_min_y": global_window[2],
                    "tile_max_y": global_window[3],
                },
                "frames": public_records,
            },
        )
        shutil.rmtree(self.TrainingProgressFramePath)
        print("Training progress video saved at {}".format(output_path))

    def GetDemo(self):
        if self.TrainingProgressFrameNumber:
            self._encode_training_progress()

    def SaveTimeCost(self, TimeCost, imagenum, stage_records):
        print("########################################################")
        payload = {key: float(value) for key, value in TimeCost.items()}
        payload.update(summarize_progressive_stage_timings(stage_records))
        payload["stage_records"] = list(stage_records)
        for key, value in payload.items():
            if key == "stage_records":
                continue
            print("{}: {}s".format(key, value))
        write_json_atomic(
            Path(self.MetricsRootPath) / "timing.json",
            {"schema_version": 1, **payload},
        )
        print("########################################################")

    def run(self):
        self.MakeDirs()
        while True:
            command = self.DataSaver_queue.get()
            if command[0] == "SaveTrainingProgress":
                self.SaveTrainingProgress(
                    command[1], command[2], command[3]
                )
            elif command[0] == "GetDemo":
                self.GetDemo()
            elif command[0] == "SaveTimeCost":
                self.SaveTimeCost(command[1], command[2], command[3])
            elif command[0] == "stop":
                break

# 运行主要函数
class On_the_Fly_GS:
    def __init__(self, args=None, lp=None, op=None, pp=None):
        if args is None:
            args, lp, op, pp = GetArgs()
        self.layout = RunLayout.from_root(args.Model_Path_Dir)
        self.layout.begin(
            source_dataset=args.Source_Path_Dir,
            repository_root=Path(__file__).resolve().parent.parent,
            command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            options={
                "render_all_cameras": bool(args.render_all_cameras),
                "render_training_progress": bool(
                    args.render_training_progress
                ),
                "training_progress_target_frames": (
                    args.training_progress_target_frames
                ),
                "training_progress_fps": args.training_progress_fps,
                "recovery_checkpoint_every_stages": (
                    args.recovery_checkpoint_every_stages
                ),
                "save_final_training_state": bool(
                    args.save_final_training_state
                ),
                "semantic_checkpoint_interval": args.semantic_checkpoint_interval,
                "semantic_online": semantic_online_required(args),
                "skip_final_evaluation": bool(args.skip_final_evaluation),
            },
        )
        self.layout.write_config(vars(args))
        self.DataPreProccesser_queue = mp.Queue()
        self.GaussianTrainer_queue = mp.Queue()
        self.DataSaver_queue = mp.Queue(maxsize=2)

        self.DataPreProccesser = DataPreProccesser(args, lp, op, pp)
        self.DataPreProccesser.DataPreProccesser_queue = self.DataPreProccesser_queue
        self.DataPreProccesser.GaussianTrainer_queue = self.GaussianTrainer_queue

        self.GaussianTrainer = GaussianTrainer(args, lp, op, pp)
        self.GaussianTrainer.DataPreProccesser_queue = self.DataPreProccesser_queue
        self.GaussianTrainer.GaussianTrainer_queue = self.GaussianTrainer_queue
        self.GaussianTrainer.DataSaver_queue = self.DataSaver_queue
        self.GaussianTrainer.ImagesAlreadyBeTrainedIterations = self.DataPreProccesser.ImagesAlreadyBeTrainedIterations
        self.GaussianTrainer.source_path_list = self.DataPreProccesser.source_path_list
        self.GaussianTrainer.model_path_list = self.DataPreProccesser.model_path_list
        copy_dataset_context(
            self.DataPreProccesser.args, self.GaussianTrainer.args
        )

        self.DataSaver = DataSaver()
        self.DataSaver.DataSaver_queue = self.DataSaver_queue
        self.DataSaver.SaveDirRootPath = str(self.layout.video_frames)
        self.DataSaver.MetricsRootPath = str(self.layout.metrics)
        self.DataSaver.VideoRootPath = str(self.layout.videos)
        self.DataSaver.RenderTrainingProgress = bool(
            self.GaussianTrainer.args.render_training_progress
        )
        self.DataSaver.TrainingProgressFPS = (
            self.GaussianTrainer.args.training_progress_fps
        )
        if self.DataSaver.RenderTrainingProgress:
            class_spec, colors, _ = resolve_semantic_render_palette(
                self.GaussianTrainer.args.semantic_data_root,
                self.GaussianTrainer.args.semantic_tdom_palette_path,
            )
            self.DataSaver.LegendEntries = [
                {
                    "id": item.class_id,
                    "name": item.name,
                    "color": list(colors[index]),
                }
                for index, item in enumerate(class_spec.classes)
            ]
        self.DataSaver.GTImageRootPath = self.GaussianTrainer.args.Source_Path_Dir
        self.DataSaver.StartFromImageNo = self.GaussianTrainer.args.StartFromImageNo

        capture_script = Path(__file__).with_name("capture_environment.py")
        subprocess.run(
            [
                sys.executable,
                str(capture_script),
                os.path.join(
                    self.GaussianTrainer.args.provenance_dir,
                    "environment_start.json",
                ),
            ],
            check=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )

        print("Start Thread...\n")
        self.GaussianTrainer_Thread = mp.Process(target=self.GaussianTrainer.run)
        self.DataSaver_Thread = mp.Process(target=self.DataSaver.run)
        workers = WorkerGroup({
            "trainer": self.GaussianTrainer_Thread,
            "output": self.DataSaver_Thread,
        }, timeout=args.worker_shutdown_timeout)
        self.DataPreProccesser.worker_group = workers
        self.DataPreProccesser.GaussianTrainer_process = self.GaussianTrainer_Thread
        self.DataPreProccesser.DataSaver_process = self.DataSaver_Thread
        try:
            self.GaussianTrainer_Thread.start()
            self.DataSaver_Thread.start()
            self.DataPreProccesser.run()
            write_json_atomic(Path(self.GaussianTrainer.args.metrics_dir) / "data_preprocessing.json",
                              mask_preparation_statistics())
            workers.stop("trainer", self.GaussianTrainer_queue)
            workers.stop("output", self.DataSaver_queue)
        except BaseException:
            workers.abort()
            raise
        finally:
            for queue in (
                self.DataPreProccesser_queue,
                self.GaussianTrainer_queue,
                self.DataSaver_queue,
            ):
                # Never wait for an undrainable feeder during error cleanup.
                queue.cancel_join_thread()
                queue.close()
        artifacts, final_stage, optimizer_iteration = self._finalize_success()
        subprocess.run(
            [
                sys.executable,
                str(capture_script),
                os.path.join(
                    self.GaussianTrainer.args.provenance_dir,
                    "environment_end.json",
                ),
            ],
            check=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        artifacts["provenance"] = "provenance"
        self.layout.set_status(
            "succeeded",
            artifacts=artifacts,
            final_stage=final_stage,
            optimizer_iteration=optimizer_iteration,
        )
        try:
            self.layout.publish_latest()
        except Exception as error:
            latest_error = self.layout.logs / "latest_publish_error.txt"
            try:
                latest_error.write_text(str(error) + "\n", encoding="utf-8")
            except OSError:
                pass
            SemanticWarning(
                "run succeeded but outputs/latest was not updated: {}".format(
                    error
                )
            )
        self._cleanup_success_work()
        print("\nAll Done!!!")

    def _cleanup_success_work(self):
        try:
            for candidate in (
                self.layout.recovery,
                self.layout.semantic_work,
                self.layout.stages,
                self.layout.cache,
            ):
                if candidate.exists():
                    shutil.rmtree(candidate)
            if self.layout.video_frames.exists():
                shutil.rmtree(self.layout.video_frames)
            if self.layout.work.exists() and not any(self.layout.work.iterdir()):
                self.layout.work.rmdir()
        except Exception as error:
            cleanup_error = self.layout.logs / "cleanup_error.txt"
            try:
                cleanup_error.write_text(str(error) + "\n", encoding="utf-8")
            except OSError:
                pass
            SemanticWarning(
                "run succeeded but temporary work cleanup was incomplete: {}".format(
                    error
                )
            )

    def _finalize_success(self):
        final_stage = int(self.DataPreProccesser.IM[-1])
        final_arguments = dict(vars(self.GaussianTrainer.args))
        final_arguments["source_path"] = str(
            Path(self.GaussianTrainer.args.Source_Path_Dir) / str(final_stage)
        )
        final_arguments["model_path"] = str(self.layout.root)
        self.layout.write_config(final_arguments)
        removed_stages = prune_intermediate_stage_outputs(
            self.layout.stages, final_stage
        )
        print(
            "Final publication removed {} intermediate stage directories.".format(
                len(removed_stages)
            )
        )
        records = {
            "model": finalize_stage_model(
                self.layout,
                final_stage,
                include_training_state=bool(
                    self.GaussianTrainer.args.save_final_training_state
                ),
            ),
            "metrics": "metrics",
            "logs": "logs",
        }
        if self.GaussianTrainer.args.render_TDOM:
            if not (self.layout.maps / "latest.json").is_file():
                raise RuntimeError("requested final TDOM map was not published")
            records["maps"] = "products/maps"
        camera_renders = self.layout.renders / "cameras"
        if self.GaussianTrainer.args.render_all_cameras:
            rendered_camera_count = validate_camera_render_outputs(self.layout)
            records["renders"] = {
                "cameras": {
                    "path": camera_renders,
                    "count": rendered_camera_count,
                }
            }
        progress_video = self.layout.videos / "training_progress.mp4"
        if self.GaussianTrainer.args.render_training_progress:
            if not progress_video.is_file():
                raise RuntimeError(
                    "requested training progress video was not published"
                )
            records["videos"] = {
                "training_progress": {
                    "path": progress_video,
                    "size_bytes": progress_video.stat().st_size,
                    "sha256": sha256_file(progress_video),
                }
            }
        if self.GaussianTrainer.args.enable_semantics:
            validate_final_semantic_outputs(
                self.layout,
                final_stage=final_stage,
                require_map=bool(self.GaussianTrainer.args.render_TDOM),
            )
            records["semantic_model"] = archive_semantic_state(self.layout)
            schema_source = self.layout.semantic_work / "validated_schema.json"
            if not schema_source.is_file():
                raise RuntimeError("validated final semantic schema is missing")
            with (self.layout.semantic_work / "final_rebuild_metrics.json").open() as stream:
                final_semantic_metadata = json.load(stream)
            if (SemanticClassSpec.load(schema_source).fingerprint
                    != final_semantic_metadata.get("class_spec_sha256")):
                raise RuntimeError("validated final semantic schema identity changed")
            if schema_source.is_file():
                schema_destination = self.layout.semantic_model / "schema.json"
                shutil.copy2(schema_source, schema_destination)
                records["semantic_model"]["schema"] = {
                    "path": schema_destination,
                    "size_bytes": schema_destination.stat().st_size,
                    "sha256": sha256_file(schema_destination),
                }
        summary = self._write_summary(final_stage)
        evaluation = summary.get("evaluation")
        if not isinstance(evaluation, dict):
            raise RuntimeError("final evaluation metadata is missing")
        optimizer_iteration = evaluation.get("iteration")
        if (
            isinstance(optimizer_iteration, bool)
            or not isinstance(optimizer_iteration, int)
            or optimizer_iteration < 0
        ):
            raise RuntimeError("final optimizer iteration is invalid")
        return (
            relative_artifact_records(self.layout, records),
            final_stage,
            optimizer_iteration,
        )

    def _write_summary(self, final_stage):
        summary = {"schema_version": 1, "final_stage": int(final_stage)}
        sources = {
            "evaluation": self.layout.metrics / "final_evaluation.json",
            "timing": self.layout.metrics / "timing.json",
            "semantic": self.layout.semantic_model / "sync_status.json",
            "semantic_rebuild": (
                self.layout.semantic_model / "final_rebuild_metrics.json"
            ),
        }
        for key, path in sources.items():
            if path.is_file():
                with path.open("r", encoding="utf-8") as stream:
                    summary[key] = json.load(stream)
        latest = self.layout.maps / "latest.json"
        if latest.is_file():
            with latest.open("r", encoding="utf-8") as stream:
                pointer = json.load(stream)
            pair_path = self.layout.maps / pointer["pair_manifest"]
            if sha256_file(pair_path) != pointer["pair_manifest_sha256"]:
                raise RuntimeError("final map pointer hash does not match release")
            with pair_path.open("r", encoding="utf-8") as stream:
                summary["map"] = json.load(stream)
        write_json_atomic(self.layout.metrics / "summary.json", summary)
        return summary

if __name__ == "__main__":
    mp.set_start_method("spawn")
    active_layout = None
    try:
        parsed_args, parsed_lp, parsed_op, parsed_pp = GetArgs()
        active_layout = RunLayout.from_root(parsed_args.Model_Path_Dir)
        Run = On_the_Fly_GS(parsed_args, parsed_lp, parsed_op, parsed_pp)
    except BaseException as error:
        if active_layout is not None and active_layout.manifest.is_file():
            if active_layout.read_manifest().get("status") == "running":
                status = (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                )
                failure_path = active_layout.logs / "failure.txt"
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                with failure_path.open("w", encoding="utf-8") as stream:
                    stream.write(traceback.format_exc())
                active_layout.set_status(
                    status,
                    failure={
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": "logs/failure.txt",
                    },
                )
        raise
