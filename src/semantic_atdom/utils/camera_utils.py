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

from semantic_atdom.scene.cameras import Camera
import numpy as np
import torch
from semantic_atdom.utils.general_utils import PILtoTorch
from semantic_atdom.utils.graphics_utils import fov2focal
from semantic_atdom.utils.pixel_grid import camera_resolution, nearest_indices

WARNED = False


def resize_camera_mask(mask, resolution):
    if mask is None or tuple(mask.shape) == (2,):
        return mask
    value = mask if isinstance(mask, torch.Tensor) else torch.from_numpy(np.asarray(mask))
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError("camera mask must have shape [H, W] or [1, H, W]")
    width, height = map(int, resolution)
    rows = torch.from_numpy(nearest_indices(value.shape[-2], height)).to(value.device)
    columns = torch.from_numpy(nearest_indices(value.shape[-1], width)).to(value.device)
    return value.index_select(-2, rows).index_select(-1, columns).bool()

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    resolution = camera_resolution(orig_w, orig_h, args.resolution, resolution_scale)
    if args.resolution == -1 and orig_w > 1600:
        global WARNED
        if not WARNED:
            print("[ INFO ] Large input images are rescaled to 1.6K; use --resolution 1 to retain their size")
            WARNED = True

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    tri_mask = resize_camera_mask(cam_info.Tri_Mask, resolution)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask, Tri_Mask=tri_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  image_path=cam_info.image_path,
                  calibration=getattr(cam_info, "calibration", None))

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    # Dataset CameraInfo and loaded Camera use different attribute names.
    # Serialize intrinsics on the loaded image grid (including RGB resizing).
    if isinstance(camera, Camera):
        width, height = camera.image_width, camera.image_height
        fov_x, fov_y = camera.FoVx, camera.FoVy
    else:
        width, height = camera.width, camera.height
        fov_x, fov_y = camera.FovX, camera.FovY
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : width,
        'height' : height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(fov_y, height),
        'fx' : fov2focal(fov_x, width)
    }
    return camera_entry
