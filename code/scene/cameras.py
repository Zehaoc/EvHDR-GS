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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 event=None,
                 hdr=None,
                 expourse_time=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.ori_R = torch.tensor(R)
        self.ori_T = torch.tensor(T)
        self.R = torch.tensor(R)
        self.T = torch.tensor(T)
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        if event is not None:
            self.event = torch.from_numpy(event)
        else:
            self.event = torch.zeros((self.image_height, self.image_width))
        if hdr is not None:
            hdr = torch.from_numpy(hdr)
            hdr = hdr.permute([2, 0, 1])
            self.hdr = hdr
        else:
            self.hdr = torch.zeros((3, self.image_height, self.image_width))
        if expourse_time is not None:
            self.expourse_time = torch.tensor([expourse_time]).cuda()
        else:
            self.expourse_time = torch.tensor([1]).cuda()
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        
        confidence_mask = torch.from_numpy(self.get_confidence_mask(self.original_image))
        confidence_mask = confidence_mask.permute([2, 0, 1])
        self.confidence_mask = confidence_mask
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        # self.transform = torch.nn.Parameter(torch.randn(4, 4))

    def get_confidence_mask(self, image):
        def BGR2YCbCr_numpy(image):
            b = image[:, :, 0]  
            g = image[:, :, 1]  
            r = image[:, :, 2]  
            yr = .299 * r + .587 * g + .114 * b
            cb = (b - yr) * .564
            cr = (r - yr) * .713
            return np.stack((yr, cb, cr), -1)
        image = image.permute([1, 2, 0]).cpu().numpy()
        Y_ldr = BGR2YCbCr_numpy(image)[:, :, 0:1]
        lamb = 0.8
        mask_ldr = (0.5 - np.maximum(np.abs(Y_ldr - 0.5), np.ones_like(Y_ldr) * (lamb - 0.5))) / (1.0 - lamb)
        mask_ldr = np.tile(mask_ldr, (1,1,3))
        return mask_ldr
    
    def update_pose(self, pose_transform_r, pose_transform_t):
        self.update_R(pose_transform_r)
        self.update_T(pose_transform_t)
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def update_R(self, pose_transform):
        norm = torch.sqrt(pose_transform[0]*pose_transform[0] + pose_transform[1]*pose_transform[1] + pose_transform[2]*pose_transform[2] + pose_transform[3]*pose_transform[3])
        pose_transform = pose_transform / norm
        w, x, y, z = pose_transform[0], pose_transform[1], pose_transform[2], pose_transform[3]
        delta_R = torch.zeros((3, 3), dtype=self.R.dtype)
        delta_R[0, 0] = 1 - 2 * (y**2 + z**2)
        delta_R[0, 1] = 2 * (x * y - z * w)
        delta_R[0, 2] = 2 * (x * z + y * w)
        delta_R[1, 0] = 2 * (x * y + z * w)
        delta_R[1, 1] = 1 - 2 * (x**2 + z**2)
        delta_R[1, 2] = 2 * (y * z - x * w)
        delta_R[2, 0] = 2 * (x * z - y * w)
        delta_R[2, 1] = 2 * (y * z + x * w)
        delta_R[2, 2] = 1 - 2 * (x**2 + y**2)        
        self.R = delta_R @ self.ori_R
        # self.R = self.ori_R
    
    def update_T(self, delta_T):
        self.T = self.ori_T + delta_T
        # self.T = self.ori_T + delta_T
        # print('new_T', self.T .shape)        
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

