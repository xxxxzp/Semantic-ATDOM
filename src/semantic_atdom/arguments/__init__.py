#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import json
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self.source_path_second = ""  # COLMAP生成的稀疏点云存储所在的位置
        self.model_path_second = ""  # 训练得到的模型存储的位置
        self.depth_map_path = ""
        self.normal_map_path = ""
        self.Image_Folder_Path = ""
        self._images = "images"
        self._resolution = -1
        self.depth_scale = 1.0
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.GetNormal = False
        self.NotGetLoad = False
        self.NoTaming = False
        self.UseDepthLoss = False
        self.UseScaleLoss = False
        self.InitialTrainingTimesSetZero = False
        self.skip_FirstSceneTraining = False
        self.skip_SecondSceneInitialization = False
        self.skip_MergeSceneTraining = False
        self.skip_GlobalOptimization = False
        self.DifferentImagesLr = False
        self.DifferentImagesDensify = False
        self.Use_Global_Optimization_densify = False
        self.Use_Global_Optimization_OpacityReset = False
        self.OnlyUseGlobalOptimization = False
        self.OnlyUseLocalOptimization = False
        self.OutputModel = False
        self.NoDebug = False
        self.GetMergeMaskQuickly = False
        self.DifferentImagesResetOpacity = False
        self.UseTrueDepthMap = False
        self.OnlyTrainNewScene = False
        self.ProgressiveModelOutput = False
        self.GetDemo = False
        self.Block = False
        self.render_RGB = False
        self.render_target = False
        self.Small_Scale_Scene = False
        self.Use_Tri_Mask = False
        self.Single = False
        self.render_TDOM = False
        self.enable_semantics = False
        self.Get_Pos = False

        # Use for Ablation
        self.No_Key = False
        self.No_Sampling = False
        self.No_Adaptive = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_load = 0.41  # NEW
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.opacity_threshold = 0.005
        self.random_background = False
        self.optimizer_type = "default"

        self.StartFromImageNo = 30
        self.EndAtImageNo = -1
        self.ContinueFromImageNo = -1
        self.ImagesNumLimitation = -1
        self.gpu = 0
        # Completed production runs retain only their final model by default.
        # Set this above zero explicitly when intermediate recovery points are
        # worth their substantial serialization time and disk cost.
        self.OutputModelInterval = 0
        self.GlobalOptimizationInterval = 1
        self.GlobalOptimizationIteration = 100
        self.progressive_identity_policy = "strict_cumulative"
        self.Source_Path_Dir = ""
        self.Model_Path_Dir = ""
        self.IterationFirstScene = 2000
        self.IterationPerMergeScene = 100
        self.FinalOptimizationIterations = 4000
        self.MergeScene_Densification_Interval = 100
        self.MergeScene_Densify_Grad_Threshold = 0.0002
        self.GaussiansNumMax = 1000000
        self.GaussianBudgetSlack = 0.05
        self.ResetOpacityUntilIter = 300000
        self.RenderAllImagesInterval = 5000
        self.InitialTrainingEvaluateInterval = 500
        self.FinalTrainingEvaluateInterval = 1000
        self.spatial_lr_scale = 0.0
        self.Mean = 200
        self.points_per_triangle = 8
        self.OriginImageHeight = 0
        self.OriginImageWidth = 0
        self.LogMaskThreshold = 0.1
        self.TDOM_FoVx = 3.0
        self.GSD_x = 0.00
        self.GSD_y = 0.00
        self.semantic_overlap_threshold = 0.4
        self.semantic_max_group_size = 4
        self.semantic_reliability_floor = 0.1
        self.semantic_temperature = 0.1
        self.semantic_boundary_bandwidth = 2.0
        self.semantic_boundary_floor = 0.25
        self.semantic_data_root = ""
        self.semantic_tdom_palette_path = ""
        self.semantic_tdom_min_coverage = 0.05
        self.semantic_tdom_min_absolute_support = 0.01
        self.semantic_tdom_min_support_ratio = 0.05
        self.semantic_tdom_render_tile_rows = 0
        self.semantic_tdom_finalize_tile_rows = 256
        self.semantic_tdom_save_diagnostics = False
        self.semantic_tdom_save_probabilities = False
        self.semantic_tdom_save_class_ids = False
        self.semantic_tdom_save_masks = False
        self.semantic_tdom_save_evidence = False
        self.semantic_tdom_save_entropy = False
        self.semantic_tdom_final_only = True
        self.semantic_final_rebuild = False
        self.semantic_tdom_max_output_bytes = 8 * 1024**3
        self.semantic_tdom_min_free_cuda_bytes = 512 * 1024**2
        self.semantic_tdom_max_gaussian_scale_ratio = 14.0
        # Optional final-product recovery for bounded holes in a known planar
        # class.  An empty class name keeps the dataset-agnostic default off.
        self.semantic_tdom_planar_fallback_class = ""
        self.semantic_tdom_planar_fallback_min_component_pixels = 1024
        self.semantic_tdom_planar_fallback_ring_radius_pixels = 15
        self.semantic_tdom_planar_fallback_min_boundary_consensus = 0.98
        self.semantic_tdom_planar_fallback_plane_context_pixels = 512
        self.semantic_tdom_planar_fallback_min_plane_gaussians = 32
        self.semantic_tdom_planar_fallback_max_plane_mad_world = 1.0
        self.semantic_tdom_planar_fallback_min_views = 5
        self.semantic_tdom_planar_fallback_min_agreement = 0.95
        self.semantic_tdom_planar_fallback_min_groups = 3
        self.semantic_map_publication_interval = 1
        # Zero disables mutable semantic recovery slots during progressive
        # stages. The final semantic model is still archived once.
        self.semantic_checkpoint_interval = 0
        self.semantic_map_tile_size = 512
        self.semantic_online_time_budget_seconds = 30.0
        self.semantic_cuda_stream_priority = 0

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    args_cmdline = parser.parse_args(cmdlne_string)
    merged_dict = {}

    if args_cmdline.model_path:
        config_path = os.path.join(
            args_cmdline.model_path, "provenance", "config.json"
        )
        try:
            with open(config_path, encoding="utf-8") as config_file:
                payload = json.load(config_file)
            if payload.get("schema_version") != 1 or not isinstance(
                payload.get("arguments"), dict
            ):
                raise ValueError("unsupported run config schema")
            merged_dict.update(payload["arguments"])
            print("Config file found: {}".format(config_path))
        except FileNotFoundError:
            pass

    if not merged_dict:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for legacy config file in", cfgfilepath)
        try:
            with open(cfgfilepath) as cfg_file:
                print("Legacy config file found: {}".format(cfgfilepath))
                args_cfgfile = eval(cfg_file.read(), {"Namespace": Namespace})
            merged_dict.update(vars(args_cfgfile))
        except (TypeError, FileNotFoundError):
            pass

    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
