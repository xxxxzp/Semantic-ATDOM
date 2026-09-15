import os
import hashlib
import random
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
from argparse import ArgumentParser, Namespace
from semantic_atdom.arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial import cKDTree
import math
import time
from semantic_atdom.scene.cameras import Camera, DummyCamera, DummyPipeline
from semantic_atdom.utils.camera_utils import camera_to_JSON
import json
from semantic_atdom.scene.dataset_readers import sceneLoadTypeCallbacks
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
from semantic_atdom.utils.graphics_utils import getWorld2View2, getProjectionMatrix
import torchvision
import torch.multiprocessing as mp
from skimage.metrics import structural_similarity as ssim
import lpips
from semantic_atdom.semantics.orchestrator import (
    AsyncProgressiveSemanticHook,
    MapPublicationRequest,
    OnlineSemanticConfig,
)
from semantic_atdom.semantics.publication import (
    MapGridSpec,
    MapPublicationPolicy,
)
from semantic_atdom.semantics.schema import SemanticClassSpec
from semantic_atdom.semantics.preflight import run_dataset_preflight
from semantic_atdom.semantics.tdom import (
    SemanticTdomConfig,
    load_semantic_tdom_palette,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

WARNED = False

def SemanticWarning(message):
    """Warnings-as-errors must never couple semantics back into geometry."""
    try:
        warnings.warn(message, RuntimeWarning)
    except Exception:
        pass

def Get_lpips(rgbs: torch.Tensor, target_rgbs: torch.Tensor, lpips_vgg):
    gt = target_rgbs.unsqueeze(0) * 2 - 1
    pred = rgbs.unsqueeze(0) * 2 - 1

    lpips_vgg_i = lpips_vgg(gt, pred, normalize=True)
    # lpips_alex_i = lpips_alex(gt, pred, normalize=True)
    # lpips_squeeze_i = lpips_squeeze(gt, pred, normalize=True)

    return {'vgg': lpips_vgg_i.item()}

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

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    '''以上是在构建参数配置器'''

    # 输出存储训练后模型的存储位置
    print("Optimizing " + args.Source_Path_Dir)
    layout = resolve_progressive_dataset_layout(
        args.Source_Path_Dir, args.Image_Folder_Path or None
    )
    args.Source_Path_Dir = str(layout.progressive_root)
    if not args.Image_Folder_Path and layout.image_root is not None:
        args.Image_Folder_Path = str(layout.image_root)
        print("Discovered shared RGB folder: " + args.Image_Folder_Path)

    return args, lp, op, pp

def TrainingPreparation(args):
    TimeCost = {}

    if args.enable_semantics and not str(args.semantic_data_root).strip():
        raise ValueError(
            "--semantic_data_root is required when --enable_semantics is set"
        )

    # 输出训练日志
    os.makedirs(args.Model_Path_Dir, exist_ok=True)

    if args.ContinueFromImageNo == -1:
        Diary = open(os.path.join(args.Model_Path_Dir, "Diary.txt"), "w")
        EvaluateDiary = open(os.path.join(args.Model_Path_Dir, "EvaluateDiary.txt"), "w")
        GaussianDiary = open(os.path.join(args.Model_Path_Dir, "GaussianDiary.txt"), "w")
    else:
        Diary = open(os.path.join(args.Model_Path_Dir, "Diary.txt"), "a")
        EvaluateDiary = open(os.path.join(args.Model_Path_Dir, "EvaluateDiary.txt"), "a")
        GaussianDiary = open(os.path.join(args.Model_Path_Dir, "GaussianDiary.txt"), "a")

    # 这个字典用于存储每一张影像已经被训练了多少次
    ImagesAlreadyBeTrainedIterations = {}

    # 一些输出上的设置
    safe_state(args.quiet)
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
    dataset_id = hashlib.sha256(
        os.path.abspath(args.Source_Path_Dir).encode("utf-8")
    ).hexdigest()[:16]
    args.dataset_cache_root = os.path.join(
        args.Model_Path_Dir, "cache", dataset_id
    )
    if args.enable_semantics:
        preflight = run_dataset_preflight(
            dataset_root=args.Source_Path_Dir,
            semantic_data_root=args.semantic_data_root,
            output_root=args.Model_Path_Dir,
            start_stage=args.StartFromImageNo,
            end_stage=args.EndAtImageNo,
            image_folder_path=(args.Image_Folder_Path or None),
        )
        args.dataset_preflight_path = str(preflight.report_path)
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
    print(IM)
    source_path_list = [os.path.join(args.Source_Path_Dir, str(im)) for im in IM]
    model_path_list = [os.path.join(args.Model_Path_Dir, str(im)) for im in IM]
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
        WholeIterations = args.IterationFirstScene + (IM[-1] - IM[0] - 1) * args.IterationPerMergeScene + \
                          int((IM[-1] - IM[0] - 2) / args.GlobalOptimizationInterval) * args.GlobalOptimizationIteration + \
                          args.FinalOptimizationIterations
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
    def __init__(self):
        super().__init__()

        self.Silence = True

        # 用于两个进程之间的信息传递
        self.DataPreProccesser_queue = None
        self.GaussianTrainer_queue = None

        # 获取参数配置器
        self.args, self.lp, self.op, self.pp = GetArgs()

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
                Image_Folder_Path=self.args.Image_Folder_Path)
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
                Image_Folder_Path=self.args.Image_Folder_Path)
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
                                                            fovX=TrainCameras[cam.image_name]["FoVx"],
                                                            fovY=TrainCameras[cam.image_name]["FoVy"]).transpose(0, 1) # 4*4
                    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                    camera_center = world_view_transform.inverse()[3, :3]

                    UpdatedTrainCamsPos[cam.image_name] = [world_view_transform, projection_matrix, full_proj_transform, camera_center]

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
                orig_w, orig_h = cam_info.image.size
                if self.args.resolution in [1, 2, 4, 8]:
                    resolution = round(orig_w / (resolution_scale * self.args.resolution)), round(
                        orig_h / (resolution_scale * self.args.resolution))
                else:  # should be a type that converts to float
                    if self.args.resolution == -1:
                        if orig_w > 1600:
                            global WARNED
                            if not WARNED and not self.Silence:
                                print(
                                    "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                                    "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                                WARNED = True
                            global_down = orig_w / 1600
                        else:
                            global_down = 1
                    else:
                        global_down = orig_w / self.args.resolution

                    scale = float(global_down) * float(resolution_scale)
                    resolution = (int(orig_w / scale), int(orig_h / scale))

                resized_image_rgb = PILtoTorch(cam_info.image, resolution)

                gt_image = resized_image_rgb[:3, ...]
                loaded_mask = None

                if resized_image_rgb.shape[1] == 4:
                    loaded_mask = resized_image_rgb[3:4, ...]

                # NewCam = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                #                 FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                #                 image=gt_image, gt_alpha_mask=loaded_mask, Tri_Mask=cam_info.Tri_Mask,
                #                 image_name=cam_info.image_name, uid=len(self.scene.train_cameras[resolution_scale]),
                #                 data_device=self.args.data_device)
                NewCam = Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                                FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                                image=gt_image, gt_alpha_mask=loaded_mask, Tri_Mask=cam_info.Tri_Mask,
                                image_name=cam_info.image_name, uid=len(TrainCameras),
                                data_device=self.args.data_device)

                # self.scene.train_cameras[resolution_scale].append(NewCam)
                # TrainCameras.append(NewCam)

                self.GetGradLogForCams([NewCam])

                with torch.no_grad():
                    # render_pkg = render(NewCam, self.gaussians, self.args, self.bg, Render_Mask=NewCam.Tri_Mask)
                    # image = render_pkg["render"]

                    # 像后端发送指令，令其渲染一张当前新影像
                    msg = ["Render_New", NewCam]
                    self.GaussianTrainer_queue.put(msg)

                    while True:
                        if not self.DataPreProccesser_queue.empty():
                            image = self.DataPreProccesser_queue.get()[0]
                            break

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
        self.GaussianTrainer_queue.put(["init"])

        StartFrom = 0 if self.args.ContinueFromImageNo == -1 else self.args.ContinueFromImageNo - self.args.StartFromImageNo
        if StartFrom > len(self.source_path_list) - 2:
            while True:
                if self.DataPreProccesser_queue.empty():
                    continue
                command = self.DataPreProccesser_queue.get()
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
                        if not self.DataPreProccesser_queue.empty():
                            if self.DataPreProccesser_queue.get()[0] == "stop":
                                return 0
                if command[0] == "stop":
                    return 0

        ProgressiveTrainingTime = StartFrom
        while True:
            # 在后端没有发送任何指令之前，不做任何行为
            if self.DataPreProccesser_queue.empty():
                continue
            else:
                Commond_from_GaussianTrainer = self.DataPreProccesser_queue.get()
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

                        # 像后端发送指令和数据，开始对新进影像的训练
                        print("DataPreProccesser: Send Information to GaussianTrainer...")
                        # msg = ["Progressive_Train", New_scene_info_traincam, NewCams, TrainCameras, scene_info, new_model_path_second, cameras_extent, self.NewImagesNum]
                        msg = ["Progressive_Train", None, NewCams, [worlds, projs, fulls, centers, names], scene_info.point_cloud, new_model_path_second, cameras_extent, self.NewImagesNum]
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

                        # 像后端发送指令和数据，开始对新进影像的训练，此时，需要注意先等后端把上一张影像完成训练之后，再进行传输
                        while True:
                            if not self.DataPreProccesser_queue.empty():
                                if self.DataPreProccesser_queue.get()[0] == "Progressive_finish":
                                    # 像后端发送指令和数据，开始对新进影像的训练
                                    print("DataPreProccesser: Send Information to GaussianTrainer...")
                                    msg = ["Progressive_Train", None, NewCams, [worlds, projs, fulls, centers, names], scene_info.point_cloud, new_model_path_second, cameras_extent, self.NewImagesNum]
                                    # msg = ["Progressive_Train"]
                                    self.GaussianTrainer_queue.put(msg)
                                break

                        ProgressiveTrainingTime += 1
                    else:
                        msg = ["Final_Refinement_Prepare"]
                        self.GaussianTrainer_queue.put(msg)

                        # 像后端发送指令和数据，开始进行最终的全局优化，此时，需要注意先等后端把上一张影像完成训练之后，再进行传输
                        while True:
                            if not self.DataPreProccesser_queue.empty():
                                if self.DataPreProccesser_queue.get()[0] == "Progressive_finish":
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
    def __init__(self):
        super().__init__()

        # 获取参数配置器
        self.args, self.lp, self.op, self.pp = GetArgs()

        self.ImagesAlreadyBeTrainedIterations = None
        self.ResetOpacityTime = 0

        # 用于两个进程之间的信息传递
        self.DataPreProccesser_queue = None
        self.GaussianTrainer_queue = None
        self.DataSaver_queue = None

        # 记录模型的已训练次数
        self.AlreadyTrainingIterations = 0
        self.StartFromTrainingIterations = 0

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

        self.TimeCost = {"GaussianInitial": 0.0, "ProgressiveTrain": 0.0, "FinalRefinement": 0.0}

        self.EvaluateDiary = None

        self.Finish = False

        # 在SLAM数据集上可能出现训练速度快于前端处理速度的现象，因此该布尔值用于表示是否已经完成“Render_New”这个操作
        self.RenderNewDone = False

        # Optional WP3 sidecar. It is deliberately outside every RGB optimizer.
        self.semantic_hook = None
        self.semantic_tdom_config = None
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
            self.args.Model_Path_Dir, "Map_Products", "map_grid.json"
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
            semantic_root = os.path.join(semantic_dataset_root, "semantic")
            palette_path = self.args.semantic_tdom_palette_path
            if not palette_path:
                candidate = os.path.join(semantic_root, "tdom_palette.json")
                if os.path.isfile(candidate):
                    palette_path = candidate
            render_palette = None
            render_palette_source = None
            if palette_path:
                try:
                    class_spec = SemanticClassSpec.load(
                        os.path.join(semantic_root, "class_spec.json")
                    )
                    render_palette = load_semantic_tdom_palette(
                        palette_path, class_spec
                    )
                    render_palette_source = palette_path
                except Exception as error:
                    SemanticWarning(
                        "Semantic TDOM palette ignored; using class_spec "
                        "colors so semantic fusion can continue: {}".format(
                            error
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
                min_margin=self.args.semantic_tdom_min_margin,
                min_top1_probability=(
                    self.args.semantic_tdom_min_top1_probability
                ),
                max_normalized_entropy=(
                    self.args.semantic_tdom_max_normalized_entropy
                ),
                max_group_disagreement=(
                    self.args.semantic_tdom_max_group_disagreement
                ),
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
            self.semantic_hook = AsyncProgressiveSemanticHook.create(
                dataset_root=self.args.Source_Path_Dir,
                output_root=os.path.join(
                    self.args.Model_Path_Dir, "semantic"
                ),
                gaussians=self.gaussians,
                config=config,
                semantic_dataset_root=semantic_dataset_root,
                publication_policy=self.semantic_publication_policy,
                resume=not force_fresh,
                final_authoritative_complete=(
                    self.args.semantic_authoritative_exact_complete
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
                    self.args.Model_Path_Dir,
                    "Map_Products",
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
        try:
            result = hook.synchronize(
                geometry_stage=self.imagenum,
                cameras=self.scene.getTrainCameras(),
                gaussians=self.gaussians,
                snapshot_root=snapshot_root,
                final=final,
                publication_request=publication_request,
            )
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

        # 为当前已经读入的影像全部创建一个梯度图
        self.GetGradLogForCams()

        # 进行一些训练上的初始化设置
        self.gaussians.training_setup(self.args)

        # 进行初始模型的训练
        time1 = time.time()
        if (not self.args.skip_FirstSceneTraining) and self.args.ContinueFromImageNo == -1:
            self.Train_Gaussians(self.args.IterationFirstScene, "Initialization")
        time2 = time.time()
        self.TimeCost["GaussianInitial"] = time2 - time1

        # WP3 synchronization point: initialization/densify/prune is complete.
        if self.args.render_TDOM and self.TDOM_Cam is None:
            with torch.no_grad():
                self._prepare_semantic_map_safely("Initialization")
                if not self.args.enable_semantics:
                    self._render_legacy_tdom()
        self._initialize_semantic_hook()
        self._run_semantic_hook("initialization")

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
                                             "FoVy": cam.FoVy}

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
        if self.args.enable_semantics and (
            self.semantic_publication_policy is None
            or (self.args.semantic_tdom_final_only and not final)
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
                    self.args.Model_Path_Dir,
                    "Map_Products",
                    "map_grid.json",
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
            "dataset_preflight": {
                "path": self.args.dataset_preflight_path,
                "sha256": self.args.dataset_preflight_sha256,
                "authoritative_exact_complete": (
                    self.args.semantic_authoritative_exact_complete
                ),
            },
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
        """Preserve the RGB-only path when semantic pairing is disabled."""
        if self.TDOM_Cam is None:
            return
        tdom = render(
            self.TDOM_Cam,
            self.gaussians,
            self.TDOM_pipeline,
            self.background,
            Render_Mask=torch.ones(1, dtype=torch.bool).cuda(),
            TDOM=True,
        )["render"]
        output = tdom_to_uint8(tdom)
        del tdom
        self.DataSaver_queue.put(["SaveTDOM", output, self.TDOM_alignment])

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

            # 随机从摄影机视点栈堆中任意取出一个影像以及其视点信息
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            # 如果当前影像异常，应当跳过并返回一次训练次数
            if viewpoint_cam.Tri_Mask.shape[0] == 2:
                self.AlreadyTrainingIterations -= 1
                continue

            # 学习率下降
            if Training_Type == "Initialization" or self.args.No_Adaptive:
                lr = self.gaussians.update_learning_rate(self.AlreadyTrainingIterations)
            else:
                lr = self.gaussians.SingleImage_Update_Lr(self.ImagesAlreadyBeTrainedIterations,
                                                          self.args, viewpoint_cam,
                                                          self.args.IterationPerMergeScene + self.args.GlobalOptimizationIteration)

            # 将球谐函数的阶数提高一阶，最多至3阶
            if self.AlreadyTrainingIterations % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # 在进行渐进式训练前，保证球谐函数已经提升至最高
            if self.gaussians.active_sh_degree < self.args.sh_degree:
                self.gaussians.oneupSHdegree()

            # 计算损失
            if Training_Type != "On_The_Fly":
                # 进行影像渲染，渲染指定视点的影像
                render_pkg = render(viewpoint_cam, self.gaussians, self.args, self.bg, Render_Mask=viewpoint_cam.Tri_Mask)
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

                self.Image_Visibility[viewpoint_cam.image_name] = visibility_filter

                # 计算损失
                if self.args.Use_Tri_Mask:
                    image = image * viewpoint_cam.Tri_Mask
                    load_distribution = load_distribution[viewpoint_cam.Tri_Mask.squeeze(0)]

                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                loss = (1.0 - self.args.lambda_dssim) * Ll1 + self.args.lambda_dssim * (1.0 - ssim_value)

                load_loss = torch.std(load_distribution)  # 计算标准差
                loss_adj = math.floor(math.log10(load_loss)) - math.floor(math.log10(loss))
                load_loss = load_loss / math.pow(10, loss_adj + 1.0)

                # loss = loss * (1 - self.args.lambda_load) + self.args.lambda_load * load_loss
            # 目前暂时采用与其他情况相同的训练策略
            else:
                # 进行影像渲染，渲染指定视点的影像
                render_pkg = render(viewpoint_cam, self.gaussians, self.args, self.bg,
                                    Render_Mask=viewpoint_cam.Tri_Mask)
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

                self.Image_Visibility[viewpoint_cam.image_name] = visibility_filter

                # 计算损失
                if self.args.Use_Tri_Mask:
                    image = image * viewpoint_cam.Tri_Mask
                    load_distribution = load_distribution[viewpoint_cam.Tri_Mask.squeeze(0)]

                gt_image = viewpoint_cam.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                loss = (1.0 - self.args.lambda_dssim) * Ll1 + self.args.lambda_dssim * (1.0 - ssim_value)

                load_loss = torch.std(load_distribution)  # 计算标准差
                loss_adj = math.floor(math.log10(load_loss)) - math.floor(math.log10(loss))
                load_loss = load_loss / math.pow(10, loss_adj + 1.0)

                # loss = loss * (1 - self.args.lambda_load) + self.args.lambda_load * load_loss

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
                # 计算log损失累计
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log

                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                if sub_iteration == iteration:
                    EvaluateRender = True
                    progress_bar.close()

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

        # 在完成这一张影像的训练之后
        if Training_Type == "On_The_Fly" and not self.args.No_Sampling:
            with torch.no_grad():
                # 进行一次高斯球剔除
                prune_mask = (self.gaussians.get_opacity < self.args.opacity_threshold).squeeze()
                self.gaussians.prune_points(prune_mask)
                print(f"Prune Gaussians Num: {prune_mask.sum()}")

        # 如果需要，进行一次影像渲染或者TDOM的生成
        if Training_Type == "On_The_Fly" and self.args.render_RGB:
            print("GaussianTrainer: Save RGB Results...")
            with torch.no_grad():
                for i in range(len(self.NewCams)):
                    render_pkg = render(self.NewCams[i], self.gaussians, self.args, self.bg, Render_Mask=self.NewCams[0].Tri_Mask)
                    image = render_pkg["render"]
                    self.DataSaver_queue.put(["SaveImage", image])
                    self.DataSaver_queue.put(["SaveGT", self.NewCams[i].image_name])
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
                self.EvaluateDiary = open(os.path.join(self.args.Model_Path_Dir, "Intermediate_Results", "eval.txt"), "w")

            if self.ChosenImageNames == []:
                validation_configs = ({'name': 'test', 'cameras': self.scene.getTestCameras()},
                                      {'name': 'train', 'cameras': self.scene.getTrainCameras()})
            else:
                validation_configs = ({'name': 'test', 'cameras': self.scene.getTestCameras()},
                                      {'name': 'train', 'cameras': [traincam for traincam in self.scene.getTrainCameras() if
                                                                    traincam.image_name in self.ChosenImageNames]})

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    lpips_test = 0.0
                    ssim_test = 0.0

                    PSNRs = []
                    ErrorImageNum = 0
                    for idx, viewpoint in enumerate(config['cameras']):
                        if viewpoint.Tri_Mask.shape[0] != 2:
                            image = torch.clamp(
                                renderFunc(viewpoint, self.gaussians, *renderArgs, Render_Mask=viewpoint.Tri_Mask)[
                                    "render"], 0.0, 1.0)
                            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                            l1_test += l1_loss(image, gt_image).mean().double()

                            PSNR = psnr(image, gt_image, viewpoint.Tri_Mask).mean().double()
                            psnr_test += PSNR
                            PSNRs.append(PSNR)

                            if self.Finish:
                                lpips_test += Get_lpips(image, gt_image, lpips_vgg)['vgg']

                                # 自动选择合适的 win_size，避免小图像报错
                                np_image = image.detach().cpu().numpy().astype(np.float32)
                                np_gt_image = gt_image.detach().cpu().numpy().astype(np.float32)

                                min_dim = min(np_image.shape[:2])
                                win_size = min(7, min_dim) if min_dim >= 7 else min_dim

                                ssim_test += ssim(np_gt_image, np_image, channel_axis=-1, data_range=1, win_size=win_size)

                                progress_bar.update(1)

                        else:
                            ErrorImageNum = ErrorImageNum + 1

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

                    self.EvaluateDiary.write("[ITER {}] Evaluating {}: L1 {} Gaussians {} PSNR {} SSIM {} LPIPS {}\n".format(self.AlreadyTrainingIterations,
                                                                                         config['name'], l1_test,
                                                                                         self.gaussians.get_xyz.shape[0],
                                                                                         psnr_test, ssim_test, lpips_test))

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
        if single:
            for i in range(self.args.IterationPerMergeScene + self.args.GlobalOptimizationIteration):
                viewpoint_stack.append(self.NewCams[0])
        else:
            for i in range(self.args.IterationPerMergeScene):
                for j in range(len(self.NewCams)):
                    viewpoint_stack.append(self.NewCams[j])

            ALL_viewpoint_stack = self.scene.getTrainCameras().copy()
            ImageIndex = [j for j in range(len(ALL_viewpoint_stack))]
            RestImageNeed = self.args.GlobalOptimizationIteration * len(self.NewCams)
            while RestImageNeed > 0:
                if len(ImageIndex) > RestImageNeed:
                    SampleIndex = random.sample(ImageIndex, RestImageNeed)
                    for i in SampleIndex:
                        viewpoint_stack.append(ALL_viewpoint_stack[i])
                    RestImageNeed = 0
                else:
                    for i in ImageIndex:
                        viewpoint_stack.append(ALL_viewpoint_stack[i])
                        RestImageNeed = RestImageNeed - 1
                        if RestImageNeed == 0:
                            break

        return viewpoint_stack

    # 为所有影像进行一次渲染并输出一些评估结果
    def Render_Evaluate_All_Images(self):
        with torch.no_grad():
            ALL_viewpoint_stack = self.scene.getTrainCameras().copy()
            All_Predicted_Images = []
            ALL_Image_Visibility_Filter = []
            ALL_Image_PSNR = []
            progress_bar = tqdm(range(len(ALL_viewpoint_stack)), desc="Rendering progress")
            for i in range(len(ALL_viewpoint_stack)):
                ImageOutputDir = os.path.join(self.args.Model_Path_Dir, "OutputImages", ALL_viewpoint_stack[i].image_name)
                PSNR_File_Path = os.path.join(ImageOutputDir, "PSNR.txt")
                os.makedirs(ImageOutputDir, exist_ok=True)
                PSNR_File = open(PSNR_File_Path, 'w')

                progress_bar.update(1)

                viewpoint_cam = ALL_viewpoint_stack[i]
                render_pkg = render(viewpoint_cam, self.gaussians, self.args, self.bg,
                                    Render_Mask=torch.ones(1, dtype=torch.bool).cuda())
                image, visibility_filter = render_pkg["render"], render_pkg["visibility_filter"]

                All_Predicted_Images.append(image)
                ALL_Image_Visibility_Filter.append(visibility_filter)

                ImageOutput = (image - image.min()) / (image.max() - image.min())
                ImageOutput = ImageOutput.permute(1, 2, 0).cpu().numpy()
                ImageOutput = (ImageOutput * 255).astype(np.uint8)
                ImageOutput = Image.fromarray(ImageOutput)
                ImageOutput.save(os.path.join(ImageOutputDir, f"PredictionImages{self.AlreadyTrainingIterations}.jpg"))

                gt_image = viewpoint_cam.original_image.cuda()
                PSNR = psnr(image, gt_image)
                PSNR = PSNR.cpu().sum().item() / 3
                ALL_Image_PSNR.append(PSNR)

                PSNR_File.write(str(self.AlreadyTrainingIterations) + f": {PSNR}, Visible_Gaussians_Num: {visibility_filter.shape[0]}\n")
                PSNR_File.close()

    # 保存当前模型
    def GS_Save(self, UseSecondPath=True):
        imagedirname = self.imagenum
        print("\n[Save At Image {}] Saving Checkpoint".format(imagedirname))
        torch.save((self.gaussians.capture(), self.AlreadyTrainingIterations), os.path.join(self.args.Model_Path_Dir, str(imagedirname), f"chkpnt{imagedirname}.pth"))

        print("\n[Save At Image {}] Saving Gaussians".format(imagedirname))
        self.scene.save(imagedirname)

    # 该函数将被送入到一个线程中，用于渐进式训练中的数据读取以及预处理
    def run(self):
        while True:
            if self.GaussianTrainer_queue.empty():
                continue
            else:
                Commond_from_DataPreProccesser = self.GaussianTrainer_queue.get()
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
                        time1 = time.time()
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
                                self.scene.train_cameras[1.0][i].world_view_transform = UpdatedTrainCamsPos[image_name][0].cuda()
                                self.scene.train_cameras[1.0][i].projection_matrix = UpdatedTrainCamsPos[image_name][1].cuda()
                                self.scene.train_cameras[1.0][i].full_proj_transform = UpdatedTrainCamsPos[image_name][2].cuda()
                                self.scene.train_cameras[1.0][i].camera_center = UpdatedTrainCamsPos[image_name][3].cuda()

                        # 将新的影像加入
                        for i in range(len(self.NewCams)):
                            self.scene.train_cameras[1.0].append(Commond_from_DataPreProccesser[2][i])

                        self.ImagesAlreadyBeTrainedIterations_Set()

                        # 向前端传递信息，表示开始读取后续的影像
                        self.SendImagePos_to_DataPreProccesser()

                    # 对扩张后的场景进行训练
                    if not self.args.skip_MergeSceneTraining:
                        self.Train_Gaussians((self.args.IterationPerMergeScene + self.args.GlobalOptimizationIteration) * NewImagesNum, "On_The_Fly")

                    time2 = time.time()
                    self.TimeCost["ProgressiveTrain"] += time2 - time1

                    # WP3 synchronization point: one progressive RGB batch ended.
                    if self.args.render_TDOM and self.TDOM_Cam is None:
                        with torch.no_grad():
                            self._prepare_semantic_map_safely("On_The_Fly")
                            if not self.args.enable_semantics:
                                self._render_legacy_tdom()
                    self._run_semantic_hook("progressive")

                    # 等待self.RenderNewDone变为True
                    while not self.RenderNewDone:
                        if not self.GaussianTrainer_queue.empty() or self.commond is not None:
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
                            elif self.commond[0] == "Final_Refinement_Prepare" or self.commond[0] == "RenderNewDone":
                                self.RenderNewDone = True
                                self.commond = None

                    # 向前端传递信息，表示这一张影像已经完成训练，可以将下一张影像的相关信息传输到后端
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

                    # WP3 closes the online group here; WP5 will rebuild exactly.
                    if self.args.render_TDOM and self.TDOM_Cam is None:
                        with torch.no_grad():
                            self._prepare_semantic_map_safely(
                                "Final_Refinement"
                            )
                            if not self.args.enable_semantics:
                                self._render_legacy_tdom()
                    self._run_semantic_hook("final_refinement", final=True)

                    self.Finish = True

                    msg = ["SaveTimeCost", self.TimeCost, self.imagenum]
                    self.DataSaver_queue.put(msg)

                    with torch.no_grad():
                        self.GS_Save(True)
                        self.Evaluate("Final_Refinement", True)

                    # 像后端传递信息
                    msg = ["GetDemo"]
                    self.DataSaver_queue.put(msg)

                    self.Render_Evaluate_All_Images()

                    # 向前端传递信息
                    msg = ["stop"]
                    self.DataPreProccesser_queue.put(msg)

                    self.EvaluateDiary.close()

# 后端，第三个线程，主要用于在不影响训练线程的情况下进行一些输出或者保存的工作
class DataSaver(mp.Process):
    # 构造函数
    def __init__(self):
        super().__init__()
        self.GTImageRootPath = None
        self.GTImagePath = None
        self.DataSaver_queue = None
        self.SaveDirRootPath = None
        self.Intermediate_GT_Path = None
        self.Intermediate_RGB_Render_Path = None
        self.Intermediate_TDOM_Render_Path = None
        self.Intermediate_TDOM_Gray_Render_Path = None
        self.SaveGTNumber = 0
        self.SaveImageNumber = 0
        self.SaveTDOMNumber = 0
        self.MaxTDOMWidth = 0
        self.MaxTDOMHeight = 0
        self.StartFromImageNo = 0

    # 指定保存位置并创建输出文件夹
    def MakeDirs(self):
        self.Intermediate_GT_Path = os.path.join(self.SaveDirRootPath, "GT")
        self.Intermediate_RGB_Render_Path = os.path.join(self.SaveDirRootPath, "Render")
        self.Intermediate_TDOM_Render_Path = os.path.join(self.SaveDirRootPath, "TDOM")
        self.Intermediate_TDOM_Gray_Render_Path = os.path.join(self.SaveDirRootPath, "TDOM_Gray")
        os.makedirs(self.SaveDirRootPath, exist_ok=True)
        os.makedirs(self.Intermediate_GT_Path, exist_ok=True)
        os.makedirs(self.Intermediate_RGB_Render_Path, exist_ok=True)
        os.makedirs(self.Intermediate_TDOM_Render_Path, exist_ok=True)
        os.makedirs(self.Intermediate_TDOM_Gray_Render_Path, exist_ok=True)

    # 保存一张RGB影像的GT
    def SaveGT(self, image_name):
        with torch.no_grad():
            self.GTImagePath = os.path.join(self.GTImageRootPath, str(self.StartFromImageNo + self.SaveGTNumber + 1), "images")
            SpanName = ".jpg"
            FromPath = os.path.join(self.GTImagePath, image_name + SpanName)
            if not os.path.exists(FromPath):
                SpanName = ".JPG"
                FromPath = os.path.join(self.GTImagePath, image_name + SpanName)
            if not os.path.exists(FromPath):
                SpanName = ".png"
                FromPath = os.path.join(self.GTImagePath, image_name + SpanName)
            if not os.path.exists(FromPath):
                print("Image No Exist!!!")
            else:
                ToPath = os.path.join(self.Intermediate_GT_Path, str(self.SaveGTNumber) + SpanName)
                shutil.copy2(FromPath, ToPath)
                self.SaveGTNumber += 1

    # 保存一张RGB影像
    def SaveImage(self, image):
        with torch.no_grad():
            ImageOutput = (image - image.min()) / (image.max() - image.min())
            ImageOutput = ImageOutput.permute(1, 2, 0).cpu().numpy()
            ImageOutput = (ImageOutput * 255).astype(np.uint8)
            ImageOutput = Image.fromarray(ImageOutput)
            ImageOutput.save(os.path.join(self.Intermediate_RGB_Render_Path, f"{self.SaveImageNumber}.jpg"))
            self.SaveImageNumber += 1

    # 保存TDOM
    def SaveTDOM(self, tdom, alignment=None):
        with torch.no_grad():
            # rendering_filepath = os.path.join(self.Intermediate_TDOM_Render_Path, f"{self.SaveTDOMNumber}.jpg")
            # torchvision.utils.save_image(tdom, rendering_filepath)
            if isinstance(tdom, torch.Tensor):
                ImageOutput = tdom_to_uint8(tdom)
            else:
                ImageOutput = np.asarray(tdom)
                if ImageOutput.ndim != 3 or ImageOutput.shape[2] != 3:
                    raise ValueError("TDOM output must have shape [H, W, 3]")
                if ImageOutput.dtype != np.uint8:
                    raise ValueError("preconverted TDOM output must be uint8")
            height, width = ImageOutput.shape[:2]
            if self.MaxTDOMWidth < width:
                self.MaxTDOMWidth = width
            if self.MaxTDOMHeight < height:
                self.MaxTDOMHeight = height
            ImageOutput = Image.fromarray(ImageOutput)
            image_path = os.path.join(
                self.Intermediate_TDOM_Render_Path,
                f"{self.SaveTDOMNumber}.jpg",
            )
            ImageOutput.save(image_path)
            if alignment is not None:
                with open(image_path, "rb") as stream:
                    image_sha256 = hashlib.sha256(stream.read()).hexdigest()
                payload = dict(alignment)
                payload.update(
                    {
                        "schema_version": 1,
                        "rgb_tdom_file": os.path.basename(image_path),
                        "rgb_tdom_sha256": image_sha256,
                    }
                )
                manifest_path = os.path.splitext(image_path)[0] + ".json"
                temporary = manifest_path + ".tmp"
                with open(temporary, "w", encoding="utf-8") as stream:
                    json.dump(
                        payload,
                        stream,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, manifest_path)
            self.SaveTDOMNumber += 1

    # 根据RGB影像的保存结果生成Demo
    def GetRGBDemo(self):
        # 获取影像的尺寸
        first_image_path = os.path.join(self.Intermediate_RGB_Render_Path, "0.jpg")
        frame = cv2.imread(first_image_path)
        height, width, layers = frame.shape

        # 定义视频编码器和输出格式
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式，例如'XVID'或'mp4v'
        fps = 4  # 每秒帧数
        Demo_PreIMAGE = cv2.VideoWriter(os.path.join(self.SaveDirRootPath, "Render.mp4"), fourcc, fps, (width, height))

        # 将每帧添加到视频中
        progress_bar = tqdm(range(0, self.SaveImageNumber + self.StartFromImageNo), desc="Get_RGB_Demo", initial=0)
        for i in range(self.SaveImageNumber + self.StartFromImageNo):
            text = f"({i + 1}/{self.SaveImageNumber + self.StartFromImageNo})"

            if i <= self.StartFromImageNo - 1:
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(black_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                Demo_PreIMAGE.write(black_frame)
            else:
                img_path = os.path.join(self.Intermediate_RGB_Render_Path, f"{i - self.StartFromImageNo}.jpg")
                frame = cv2.imread(img_path)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                Demo_PreIMAGE.write(frame)
            progress_bar.update(1)

        # 释放资源
        Demo_PreIMAGE.release()
        cv2.destroyAllWindows()

        print(f"RGB_Demo save at {os.path.join(self.SaveDirRootPath, 'Render.mp4')}")

    # 根据TDOM的保存结果生成Demo
    def GetTDOMDemo(self):
        scale = 0.5
        height = int(self.MaxTDOMHeight * scale)
        width = int(self.MaxTDOMWidth * scale)
        new_size = (width, height)

        # 定义视频编码器和输出格式
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式，例如'XVID'或'mp4v'
        fps = 4  # 每秒帧数
        Demo_TDOM = cv2.VideoWriter(os.path.join(self.SaveDirRootPath, "TDOM.mp4"), fourcc, fps, (width, height))

        # 将每帧添加到视频中
        progress_bar = tqdm(range(0, self.SaveTDOMNumber + self.StartFromImageNo), desc="Get_TDOM_Demo", initial=0)
        for i in range(self.SaveTDOMNumber + self.StartFromImageNo):
            text = f"TDOM: ({i + 1}/{self.SaveTDOMNumber + self.StartFromImageNo})"

            if i <= self.StartFromImageNo - 1:
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(black_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                Demo_TDOM.write(black_frame)
            else:
                img_path = os.path.join(self.Intermediate_TDOM_Render_Path, f"{i - self.StartFromImageNo}.jpg")
                expand_image_gray(
                    img_path=img_path,
                    save_path=os.path.join(self.Intermediate_TDOM_Gray_Render_Path, f"{i - self.StartFromImageNo}.jpg"),
                    width0=self.MaxTDOMWidth,
                    height0=self.MaxTDOMHeight,
                    gray_value=128  # 中性灰
                )
                frame = cv2.imread(os.path.join(self.Intermediate_TDOM_Gray_Render_Path, f"{i - self.StartFromImageNo}.jpg"))
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                Demo_TDOM.write(frame)
            progress_bar.update(1)

        # 释放资源
        Demo_TDOM.release()
        cv2.destroyAllWindows()

        print(f"TDOM_Demo save at {os.path.join(self.SaveDirRootPath, 'TDOM.mp4')}")

    # 输出时间消耗
    def SaveTimeCost(self, TimeCost, imagenum):
        outputfile = open(os.path.join(self.SaveDirRootPath, "TimeCost.txt"), "w")
        print("########################################################")
        keys = list(TimeCost.keys())
        for key in keys:
            print(f"{key}: {TimeCost[key]}s")
            outputfile.write(f"{key}: {TimeCost[key]}s\n")

            if key == "ProgressiveTrain":
                print(f"TimePerImage: {TimeCost[key] / imagenum }s")
                outputfile.write(f"TimePerImage: {TimeCost[key] / imagenum}s\n")
        outputfile.close()
        print("########################################################")

    # 运行
    def run(self):
        self.MakeDirs()
        while True:
            if self.DataSaver_queue.empty():
                continue
            else:
                Commond_from_GaussianTrainer = self.DataSaver_queue.get()
                if Commond_from_GaussianTrainer[0] == "SaveImage":
                    self.SaveImage(Commond_from_GaussianTrainer[1])
                elif Commond_from_GaussianTrainer[0] == "SaveGT":
                    self.SaveGT(Commond_from_GaussianTrainer[1])
                elif Commond_from_GaussianTrainer[0] == "SaveTDOM":
                    alignment = (
                        Commond_from_GaussianTrainer[2]
                        if len(Commond_from_GaussianTrainer) > 2
                        else None
                    )
                    self.SaveTDOM(
                        Commond_from_GaussianTrainer[1], alignment
                    )
                elif Commond_from_GaussianTrainer[0] == "GetDemo":
                    if self.SaveImageNumber > 0:
                        self.GetRGBDemo()
                    if self.SaveTDOMNumber > 0:
                        self.GetTDOMDemo()
                elif Commond_from_GaussianTrainer[0] == "SaveTimeCost":
                    TimeCost = Commond_from_GaussianTrainer[1]
                    self.SaveTimeCost(TimeCost, Commond_from_GaussianTrainer[2])
                elif Commond_from_GaussianTrainer[0] == "stop":
                    break

# 运行主要函数
class On_the_Fly_GS:
    def __init__(self):
        self.DataPreProccesser_queue = mp.Queue()
        self.GaussianTrainer_queue = mp.Queue()
        self.DataSaver_queue = mp.Queue()

        self.DataPreProccesser = DataPreProccesser()
        self.DataPreProccesser.DataPreProccesser_queue = self.DataPreProccesser_queue
        self.DataPreProccesser.GaussianTrainer_queue = self.GaussianTrainer_queue

        self.GaussianTrainer = GaussianTrainer()
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
        self.DataSaver.SaveDirRootPath = os.path.join(self.GaussianTrainer.args.Model_Path_Dir, "Intermediate_Results")
        self.DataSaver.GTImageRootPath = self.GaussianTrainer.args.Source_Path_Dir
        self.DataSaver.StartFromImageNo = self.GaussianTrainer.args.StartFromImageNo

        print("Start Thread...\n")
        self.GaussianTrainer_Thread = mp.Process(target=self.GaussianTrainer.run)
        self.GaussianTrainer_Thread.start()
        self.DataSaver_Thread = mp.Process(target=self.DataSaver.run)
        self.DataSaver_Thread.start()
        self.DataPreProccesser.run()

        self.GaussianTrainer_queue.put(["stop"])
        self.GaussianTrainer_Thread.join()
        self.DataSaver_queue.put(["stop"])
        self.DataSaver_Thread.join()
        print("\nAll Done!!!")

if __name__ == "__main__":
    mp.set_start_method("spawn")

    Run = On_the_Fly_GS()
