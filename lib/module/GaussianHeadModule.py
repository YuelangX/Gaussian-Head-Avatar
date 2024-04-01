import torch
from torch import nn
from einops import rearrange
import tqdm
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.transforms import so3_exponential_map
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion
from simple_knn._C import distCUDA2

from lib.network.MLP import MLP
from lib.network.PositionalEmbedding import get_embedder
from lib.utils.general_utils import inverse_sigmoid


class GaussianHeadModule(nn.Module):
    def __init__(self, cfg, xyz, feature, landmarks_3d_neutral, add_mouth_points=False):
        super(GaussianHeadModule, self).__init__()

        if add_mouth_points and cfg.num_add_mouth_points > 0:
            mouth_keypoints = landmarks_3d_neutral[48:66]
            mouth_center = torch.mean(mouth_keypoints, dim=0, keepdim=True)
            mouth_center[:, 2] = mouth_keypoints[:, 2].min()
            max_dist = (mouth_keypoints - mouth_center).abs().max(0)[0]
            points_add = (torch.rand([cfg.num_add_mouth_points, 3]) - 0.5) * 1.6 * max_dist + mouth_center
        
            xyz = torch.cat([xyz, points_add])
            feature = torch.cat([feature, torch.zeros([cfg.num_add_mouth_points, feature.shape[1]])])

        self.xyz = nn.Parameter(xyz)
        self.feature = nn.Parameter(feature)
        self.register_buffer('landmarks_3d_neutral', landmarks_3d_neutral)

        dist2 = torch.clamp_min(distCUDA2(self.xyz.cuda()), 0.0000001).cpu()
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        self.scales = nn.Parameter(scales)

        rots = torch.zeros((xyz.shape[0], 4), device=xyz.device)
        rots[:, 0] = 1
        self.rotation = nn.Parameter(rots)

        self.opacity = nn.Parameter(inverse_sigmoid(0.3 * torch.ones((xyz.shape[0], 1))))

        self.exp_color_mlp = MLP(cfg.exp_color_mlp, last_op=None)
        self.pose_color_mlp = MLP(cfg.pose_color_mlp, last_op=None)
        self.exp_attributes_mlp = MLP(cfg.exp_attributes_mlp, last_op=None)
        self.pose_attributes_mlp = MLP(cfg.pose_attributes_mlp, last_op=None)
        self.exp_deform_mlp = MLP(cfg.exp_deform_mlp, last_op=nn.Tanh())
        self.pose_deform_mlp = MLP(cfg.pose_deform_mlp, last_op=nn.Tanh())

        self.pos_embedding, _ = get_embedder(cfg.pos_freq)
        
        self.exp_coeffs_dim = cfg.exp_coeffs_dim
        self.dist_threshold_near = cfg.dist_threshold_near
        self.dist_threshold_far = cfg.dist_threshold_far
        self.deform_scale = cfg.deform_scale
        self.attributes_scale = cfg.attributes_scale
    
    def generate(self, data):
        B = data['exp_coeff'].shape[0]

        xyz = self.xyz.unsqueeze(0).repeat(B, 1, 1)
        feature = torch.tanh(self.feature).unsqueeze(0).repeat(B, 1, 1)

        dists, _, _ = knn_points(xyz, self.landmarks_3d_neutral.unsqueeze(0).repeat(B, 1, 1))
        exp_weights = torch.clamp((self.dist_threshold_far - dists) / (self.dist_threshold_far - self.dist_threshold_near), 0.0, 1.0)
        pose_weights = 1 - exp_weights
        exp_controlled = (dists < self.dist_threshold_far).squeeze(-1)
        pose_controlled = (dists > self.dist_threshold_near).squeeze(-1)

        color = torch.zeros([B, xyz.shape[1], self.exp_color_mlp.dims[-1]], device=xyz.device)
        delta_xyz = torch.zeros_like(xyz, device=xyz.device)
        delta_attributes = torch.zeros([B, xyz.shape[1], self.scales.shape[1] + self.rotation.shape[1] + self.opacity.shape[1]], device=xyz.device)
        for b in range(B):
            feature_exp_controlled = feature[b, exp_controlled[b], :]
            exp_color_input = torch.cat([feature_exp_controlled.t(), 
                                         data['exp_coeff'][b].unsqueeze(-1).repeat(1, feature_exp_controlled.shape[0])], 0)[None]
            exp_color = self.exp_color_mlp(exp_color_input)[0].t()
            color[b, exp_controlled[b], :] += exp_color * exp_weights[b, exp_controlled[b], :]

            feature_pose_controlled = feature[b, pose_controlled[b], :]
            pose_color_input = torch.cat([feature_pose_controlled.t(), 
                                               self.pos_embedding(data['pose'][b]).unsqueeze(-1).repeat(1, feature_pose_controlled.shape[0])], 0)[None]
            pose_color = self.pose_color_mlp(pose_color_input)[0].t()
            color[b, pose_controlled[b], :] += pose_color * pose_weights[b, pose_controlled[b], :]

            exp_attributes_input = exp_color_input
            exp_delta_attributes = self.exp_attributes_mlp(exp_attributes_input)[0].t()
            delta_attributes[b, exp_controlled[b], :] += exp_delta_attributes * exp_weights[b, exp_controlled[b], :]

            pose_attributes_input = pose_color_input
            pose_attributes = self.pose_attributes_mlp(pose_attributes_input)[0].t()
            delta_attributes[b, pose_controlled[b], :] += pose_attributes * pose_weights[b, pose_controlled[b], :]


            xyz_exp_controlled = xyz[b, exp_controlled[b], :]
            exp_deform_input = torch.cat([self.pos_embedding(xyz_exp_controlled).t(), 
                                          data['exp_coeff'][b].unsqueeze(-1).repeat(1, xyz_exp_controlled.shape[0])], 0)[None]
            exp_deform = self.exp_deform_mlp(exp_deform_input)[0].t()
            delta_xyz[b, exp_controlled[b], :] += exp_deform * exp_weights[b, exp_controlled[b], :]

            xyz_pose_controlled = xyz[b, pose_controlled[b], :]
            pose_deform_input = torch.cat([self.pos_embedding(xyz_pose_controlled).t(), 
                                           self.pos_embedding(data['pose'][b]).unsqueeze(-1).repeat(1, xyz_pose_controlled.shape[0])], 0)[None]
            pose_deform = self.pose_deform_mlp(pose_deform_input)[0].t()
            delta_xyz[b, pose_controlled[b], :] += pose_deform * pose_weights[b, pose_controlled[b], :]

        xyz = xyz + delta_xyz * self.deform_scale

        delta_scales = delta_attributes[:, :, 0:3]
        scales = self.scales.unsqueeze(0).repeat(B, 1, 1) + delta_scales * self.attributes_scale
        scales = torch.exp(scales)

        delta_rotation = delta_attributes[:, :, 3:7]
        rotation = self.rotation.unsqueeze(0).repeat(B, 1, 1) + delta_rotation * self.attributes_scale
        rotation = torch.nn.functional.normalize(rotation)

        delta_opacity = delta_attributes[:, :, 7:8]
        opacity = self.opacity.unsqueeze(0).repeat(B, 1, 1) + delta_opacity * self.attributes_scale
        opacity = torch.sigmoid(opacity)

        if 'pose' in data:
            R = so3_exponential_map(data['pose'][:, :3])
            T = data['pose'][:, None, 3:]
            S = data['scale'][:, :, None]
            xyz = torch.bmm(xyz * S, R.permute(0, 2, 1)) + T

            rotation_matrix = quaternion_to_matrix(rotation)
            rotation_matrix = rearrange(rotation_matrix, 'b n x y -> (b n) x y')
            R = rearrange(R.unsqueeze(1).repeat(1, rotation.shape[1], 1, 1), 'b n x y -> (b n) x y')
            rotation_matrix = rearrange(torch.bmm(R, rotation_matrix), '(b n) x y -> b n x y', b=B)
            rotation = matrix_to_quaternion(rotation_matrix)

            scales = scales * S

        data['exp_deform'] = exp_deform
        data['xyz'] = xyz
        data['color'] = color
        data['scales'] = scales
        data['rotation'] = rotation
        data['opacity'] = opacity
        return data
