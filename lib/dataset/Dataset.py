import torch
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import Dataset
import numpy as np
import glob
import math
import os
import random
import cv2
from skimage import io
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.transforms import so3_exponential_map

from lib.utils.graphics_utils import getWorld2View2, getProjectionMatrix


def CropImage(left_up, crop_size, image=None, K=None):
    crop_size = np.array(crop_size).astype(np.int32)
    left_up = np.array(left_up).astype(np.int32)

    if not K is None:
        K[0:2,2] = K[0:2,2] - np.array(left_up)

    if not image is None:
        if left_up[0] < 0:
            image_left = np.zeros([image.shape[0], -left_up[0], image.shape[2]], dtype=np.uint8)
            image = np.hstack([image_left, image])
            left_up[0] = 0
        if left_up[1] < 0:
            image_up = np.zeros([-left_up[1], image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.vstack([image_up, image])
            left_up[1] = 0
        if crop_size[0] + left_up[0] > image.shape[1]:
            image_right = np.zeros([image.shape[0], crop_size[0] + left_up[0] - image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.hstack([image, image_right])
        if crop_size[1] + left_up[1] > image.shape[0]:
            image_down = np.zeros([crop_size[1] + left_up[1] - image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.vstack([image, image_down])

        image = image[left_up[1]:left_up[1]+crop_size[1], left_up[0]:left_up[0]+crop_size[0], :]

    return image, K


def ResizeImage(target_size, source_size, image=None, K=None):
    if not K is None:
        K[0,:] = (target_size[0] / source_size[0]) * K[0,:]
        K[1,:] = (target_size[1] / source_size[1]) * K[1,:]

    if not image is None:
        image = cv2.resize(image, dsize=target_size)
    return image, K


class MeshDataset(Dataset):

    def __init__(self, cfg):
        super(MeshDataset, self).__init__()

        self.dataroot = cfg.dataroot
        self.camera_ids = cfg.camera_ids
        self.original_resolution = cfg.original_resolution
        self.resolution = cfg.resolution
        self.num_sample_view = cfg.num_sample_view

        self.samples = []

        image_folder = os.path.join(self.dataroot, 'images')
        param_folder = os.path.join(self.dataroot, 'params')
        camera_folder = os.path.join(self.dataroot, 'cameras')
        frames = os.listdir(image_folder)
        
        self.num_exp_id = 0
        for frame in frames:
            image_paths = [os.path.join(image_folder, frame, 'image_lowres_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            mask_paths = [os.path.join(image_folder, frame, 'mask_lowres_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            visible_paths = [os.path.join(image_folder, frame, 'visible_lowres_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            camera_paths = [os.path.join(camera_folder, frame, 'camera_%s.npz' % camera_id) for camera_id in self.camera_ids]
            param_path = os.path.join(param_folder, frame, 'params.npz')
            landmarks_3d_path = os.path.join(param_folder, frame, 'lmk_3d.npy')
            vertices_path = os.path.join(param_folder, frame, 'vertices.npy')

            sample = (image_paths, mask_paths, visible_paths, camera_paths, param_path, landmarks_3d_path, vertices_path, self.num_exp_id)
            self.samples.append(sample)
            self.num_exp_id += 1
                                  
        init_landmarks_3d = torch.from_numpy(np.load(os.path.join(param_folder, frames[0], 'lmk_3d.npy'))).float()
        init_vertices = torch.from_numpy(np.load(os.path.join(param_folder, frames[0], 'vertices.npy'))).float()
        init_landmarks_3d = torch.cat([init_landmarks_3d, init_vertices[::100]], 0)

        param = np.load(os.path.join(param_folder, frames[0], 'params.npz'))
        pose = torch.from_numpy(param['pose'][0]).float()
        R = so3_exponential_map(pose[None, :3])[0]
        T = pose[None, 3:]
        S = torch.from_numpy(param['scale']).float()
        self.init_landmarks_3d_neutral = (torch.matmul(init_landmarks_3d- T, R)) / S


    def get_item(self, index):
        data = self.__getitem__(index)
        return data
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        images = []
        masks = []
        visibles = []
        views = random.sample(range(len(self.camera_ids)), self.num_sample_view)
        for view in views:
            image_path = sample[0][view]
            image = cv2.resize(io.imread(image_path), (self.resolution, self.resolution))
            image = torch.from_numpy(image / 255).permute(2, 0, 1).float()
            images.append(image)

            mask_path = sample[1][view]
            mask = cv2.resize(io.imread(mask_path), (self.resolution, self.resolution))[:, :, 0:1]
            mask = torch.from_numpy(mask / 255).permute(2, 0, 1).float()
            masks.append(mask)

            visible_path = sample[2][view]
            if os.path.exists(visible_path):
                visible = cv2.resize(io.imread(visible_path), (self.resolution, self.resolution))[:, :, 0:1]
                visible = torch.from_numpy(visible / 255).permute(2, 0, 1).float()
            else:
                visible = torch.ones_like(image)
            visibles.append(visible)

        images = torch.stack(images)
        masks = torch.stack(masks)
        images = images * masks
        visibles = torch.stack(visibles)

        cameras = [np.load(sample[3][view]) for view in views]
        intrinsics = torch.stack([torch.from_numpy(camera['intrinsic']).float() for camera in cameras])
        extrinsics = torch.stack([torch.from_numpy(camera['extrinsic']).float() for camera in cameras])
        intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * 2 / self.original_resolution
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * 2 / self.original_resolution - 1
        intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * 2 / self.original_resolution
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * 2 / self.original_resolution - 1

        param_path = sample[4]
        param = np.load(param_path)
        pose = torch.from_numpy(param['pose'][0]).float()
        scale = torch.from_numpy(param['scale']).float()
        exp_coeff = torch.from_numpy(param['exp_coeff'][0]).float()
        
        landmarks_3d_path = sample[5]
        landmarks_3d = torch.from_numpy(np.load(landmarks_3d_path)).float()
        vertices_path = sample[6]
        vertices = torch.from_numpy(np.load(vertices_path)).float()
        landmarks_3d = torch.cat([landmarks_3d, vertices[::100]], 0)

        exp_id = sample[7]

        return {
                'images': images,
                'masks': masks,
                'visibles': visibles,
                'pose': pose,
                'scale': scale,
                'exp_coeff': exp_coeff,
                'landmarks_3d': landmarks_3d,
                'intrinsics': intrinsics,
                'extrinsics': extrinsics,
                'exp_id': exp_id}

    def __len__(self):
        return len(self.samples)




class GaussianDataset(Dataset):

    def __init__(self, cfg):
        super(GaussianDataset, self).__init__()

        self.dataroot = cfg.dataroot
        self.camera_ids = cfg.camera_ids
        self.original_resolution = cfg.original_resolution
        self.resolution = cfg.resolution

        self.samples = []

        image_folder = os.path.join(self.dataroot, 'images')
        param_folder = os.path.join(self.dataroot, 'params')
        camera_folder = os.path.join(self.dataroot, 'cameras')
        frames = os.listdir(image_folder)
        
        self.num_exp_id = 0
        for frame in frames:
            image_paths = [os.path.join(image_folder, frame, 'image_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            mask_paths = [os.path.join(image_folder, frame, 'mask_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            visible_paths = [os.path.join(image_folder, frame, 'visible_%s.jpg' % camera_id) for camera_id in self.camera_ids]
            camera_paths = [os.path.join(camera_folder, frame, 'camera_%s.npz' % camera_id) for camera_id in self.camera_ids]
            param_path = os.path.join(param_folder, frame, 'params.npz')
            landmarks_3d_path = os.path.join(param_folder, frame, 'lmk_3d.npy')
            vertices_path = os.path.join(param_folder, frame, 'vertices.npy')

            sample = (image_paths, mask_paths, visible_paths, camera_paths, param_path, landmarks_3d_path, vertices_path, self.num_exp_id)
            self.samples.append(sample)
            self.num_exp_id += 1

    def get_item(self, index):
        data = self.__getitem__(index)
        return data
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        view = random.sample(range(len(self.camera_ids)), 1)[0]

        image_path = sample[0][view]
        image = cv2.resize(io.imread(image_path), (self.original_resolution, self.original_resolution)) / 255
        mask_path = sample[1][view]
        mask = cv2.resize(io.imread(mask_path), (self.original_resolution, self.original_resolution))[:, :, 0:1] / 255
        image = image * mask + (1 - mask)

        visible_path = sample[2][view]
        if os.path.exists(visible_path):
            visible = cv2.resize(io.imread(visible_path), (self.original_resolution, self.original_resolution))[:, :, 0:1] / 255
        else:
            visible = np.ones_like(image)

        camera = np.load(sample[3][view])
        extrinsic = torch.from_numpy(camera['extrinsic']).float()
        R = extrinsic[:3,:3].t()
        T = extrinsic[:3, 3]

        intrinsic = camera['intrinsic']
        if np.abs(intrinsic[0, 2] - self.original_resolution / 2) > 1 or np.abs(intrinsic[1, 2] - self.original_resolution / 2) > 1:
            left_up = np.around(intrinsic[0:2, 2] - np.array([self.original_resolution / 2, self.original_resolution / 2])).astype(np.int32)
            _, intrinsic = CropImage(left_up, (self.original_resolution, self.original_resolution), K=intrinsic)
            image, _ = CropImage(left_up, (self.original_resolution, self.original_resolution), image=image)
            mask, _ = CropImage(left_up, (self.original_resolution, self.original_resolution), image=mask)
            visible, _ = CropImage(left_up, (self.original_resolution, self.original_resolution), image=visible)

        intrinsic[0, 0] = intrinsic[0, 0] * 2 / self.original_resolution
        intrinsic[0, 2] = intrinsic[0, 2] * 2 / self.original_resolution - 1
        intrinsic[1, 1] = intrinsic[1, 1] * 2 / self.original_resolution
        intrinsic[1, 2] = intrinsic[1, 2] * 2 / self.original_resolution - 1
        intrinsic = torch.from_numpy(intrinsic).float()

        image = torch.from_numpy(cv2.resize(image, (self.resolution, self.resolution))).permute(2, 0, 1).float()
        mask = torch.from_numpy(cv2.resize(mask, (self.resolution, self.resolution)))[None].float()
        visible = torch.from_numpy(cv2.resize(visible, (self.resolution, self.resolution)))[None].float()
        image_coarse = F.interpolate(image[None], scale_factor=0.25)[0]
        mask_coarse = F.interpolate(mask[None], scale_factor=0.25)[0]
        visible_coarse = F.interpolate(visible[None], scale_factor=0.25)[0]

        fovx = 2 * math.atan(1 / intrinsic[0, 0])
        fovy = 2 * math.atan(1 / intrinsic[1, 1])

        world_view_transform = torch.tensor(getWorld2View2(R.numpy(), T.numpy())).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=fovx, fovY=fovy).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        param_path = sample[4]
        param = np.load(param_path)
        pose = torch.from_numpy(param['pose'][0]).float()
        scale = torch.from_numpy(param['scale']).float()
        exp_coeff = torch.from_numpy(param['exp_coeff'][0]).float()
        
        landmarks_3d_path = sample[5]
        landmarks_3d = torch.from_numpy(np.load(landmarks_3d_path)).float()
        vertices_path = sample[6]
        vertices = torch.from_numpy(np.load(vertices_path)).float()
        landmarks_3d = torch.cat([landmarks_3d, vertices[::100]], 0)
        
        exp_id = torch.tensor(sample[7]).long()

        return {
                'images': image,
                'masks': mask,
                'visibles': visible,
                'images_coarse': image_coarse,
                'masks_coarse': mask_coarse,
                'visibles_coarse': visible_coarse,
                'pose': pose,
                'scale': scale,
                'exp_coeff': exp_coeff,
                'landmarks_3d': landmarks_3d,
                'exp_id': exp_id,
                'extrinsics': extrinsic,
                'intrinsics': intrinsic,
                'fovx': fovx,
                'fovy': fovy,
                'world_view_transform': world_view_transform,
                'projection_matrix': projection_matrix,
                'full_proj_transform': full_proj_transform,
                'camera_center': camera_center}

    def __len__(self):
        return len(self.samples)
    


class ReenactmentDataset(Dataset):

    def __init__(self, cfg):
        super(ReenactmentDataset, self).__init__()

        self.dataroot = cfg.dataroot
        self.original_resolution = cfg.original_resolution
        self.resolution = cfg.resolution
        self.freeview = cfg.freeview

        self.Rot_z = torch.eye(3)
        self.Rot_z[0,0] = -1.0
        self.Rot_z[1,1] = -1.0

        self.samples = []
        image_paths = sorted(glob.glob(os.path.join(self.dataroot, cfg.image_files)))
        param_paths = sorted(glob.glob(os.path.join(self.dataroot, cfg.param_files)))
        assert len(image_paths) == len(param_paths)
        
        self.samples = []
        for i, image_path in enumerate(image_paths):
            param_path = param_paths[i]
            if os.path.exists(image_path) and os.path.exists(param_path):
                sample = (image_path, param_path)
                self.samples.append(sample)

        if os.path.exists(cfg.pose_code_path):
            self.pose_code = torch.from_numpy(np.load(cfg.pose_code_path)['pose'][0]).float()
        else:
            self.pose_code = None


        self.extrinsic = torch.tensor([[1.0000,  0.0000,  0.0000,  0.0000],
                                       [0.0000, -1.0000,  0.0000,  0.0000],
                                       [0.0000,  0.0000, -1.0000,  1.0000]]).float()
        self.intrinsic = torch.tensor([[self.original_resolution * 3.5,   0.0000e+00,                     self.original_resolution / 2],
                                       [0.0000e+00,                     self.original_resolution * 3.5,   self.original_resolution / 2],
                                       [0.0000e+00,                     0.0000e+00,                       1.0000e+00]]).float()
        if os.path.exists(cfg.camera_path):
            camera = np.load(cfg.camera_path)
            self.extrinsic = torch.from_numpy(camera['extrinsic']).float()
            if not self.freeview:
                self.intrinsic = torch.from_numpy(camera['intrinsic']).float()
            
        self.R = self.extrinsic[:3,:3].t()
        self.T = self.extrinsic[:3, 3]

        self.intrinsic[0, 0] = self.intrinsic[0, 0] * 2 / self.original_resolution
        self.intrinsic[0, 2] = self.intrinsic[0, 2] * 2 / self.original_resolution - 1
        self.intrinsic[1, 1] = self.intrinsic[1, 1] * 2 / self.original_resolution
        self.intrinsic[1, 2] = self.intrinsic[1, 2] * 2 / self.original_resolution - 1

        self.fovx = 2 * math.atan(1 / self.intrinsic[0, 0])
        self.fovy = 2 * math.atan(1 / self.intrinsic[1, 1])

        self.world_view_transform = torch.tensor(getWorld2View2(self.R.numpy(), self.T.numpy())).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100, fovX=self.fovx, fovY=self.fovy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def update_camera(self, index):
        elev = math.sin(index / 20) * 8 + 0
        azim = math.cos(index / 20) * 45 - 0
        R, T = look_at_view_transform(dist=1.2, elev=elev, azim=azim, at=((0.0, 0.0, 0.05),))
        R = torch.matmul(self.Rot_z, R[0].t())
        self.extrinsic = torch.cat([R, T.t()], -1)

        self.R = self.extrinsic[:3,:3].t()
        self.T = self.extrinsic[:3, 3]

        self.world_view_transform = torch.tensor(getWorld2View2(self.R.numpy(), self.T.numpy())).transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_item(self, index):
        data = self.__getitem__(index)
        return data
    
    def __getitem__(self, index):
        if self.freeview:
            self.update_camera(index)

        sample = self.samples[index]
        
        image_path = sample[0]
        image = torch.from_numpy(cv2.resize(io.imread(image_path), (self.resolution, self.resolution)) / 255).permute(2, 0, 1).float()

        param_path = sample[1]
        param = np.load(param_path)
        pose = torch.from_numpy(param['pose'][0]).float()
        scale = torch.from_numpy(param['scale']).float()
        exp_coeff = torch.from_numpy(param['exp_coeff'][0]).float()

        if self.pose_code is not None:
            pose_code = self.pose_code
        else:
            pose_code = pose
        
        return {
                'images': image,
                'pose': pose,
                'scale': scale,
                'exp_coeff': exp_coeff,
                'pose_code': pose_code,
                'extrinsics': self.extrinsic,
                'intrinsics': self.intrinsic,
                'fovx': self.fovx,
                'fovy': self.fovy,
                'world_view_transform': self.world_view_transform,
                'projection_matrix': self.projection_matrix,
                'full_proj_transform': self.full_proj_transform,
                'camera_center': self.camera_center}

    def __len__(self):
        return len(self.samples)