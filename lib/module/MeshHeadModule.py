import torch
from torch import nn
import numpy as np
import kaolin
import tqdm
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.transforms import so3_exponential_map

from lib.network.MLP import MLP
from lib.network.PositionalEmbedding import get_embedder

from lib.utils.dmtet_utils import marching_tetrahedra

class MeshHeadModule(nn.Module):
    def __init__(self, cfg, init_landmarks_3d_neutral):
        super(MeshHeadModule, self).__init__()
        
        self.geo_mlp = MLP(cfg.geo_mlp, last_op=nn.Tanh())
        self.exp_color_mlp = MLP(cfg.exp_color_mlp, last_op=nn.Sigmoid())
        self.pose_color_mlp = MLP(cfg.pose_color_mlp, last_op=nn.Sigmoid())
        self.exp_deform_mlp = MLP(cfg.exp_deform_mlp, last_op=nn.Tanh())
        self.pose_deform_mlp = MLP(cfg.pose_deform_mlp, last_op=nn.Tanh())

        self.landmarks_3d_neutral = nn.Parameter(init_landmarks_3d_neutral)

        self.pos_embedding, _ = get_embedder(cfg.pos_freq)

        self.model_bbox = cfg.model_bbox
        self.dist_threshold_near = cfg.dist_threshold_near
        self.dist_threshold_far = cfg.dist_threshold_far
        self.deform_scale = cfg.deform_scale

        tets_data = np.load('assets/tets_data.npz')
        self.register_buffer('tet_verts', torch.from_numpy(tets_data['tet_verts']))
        self.register_buffer('tets', torch.from_numpy(tets_data['tets']))
        self.grid_res = 128

        if cfg.subdivide:
            self.subdivide()

    def geometry(self, geo_input):
        pred = self.geo_mlp(geo_input)
        return pred

    def exp_color(self, color_input):
        verts_color = self.exp_color_mlp(color_input)
        return verts_color
    
    def pose_color(self, color_input):
        verts_color = self.pose_color_mlp(color_input)
        return verts_color
    
    def exp_deform(self, deform_input):
        deform = self.exp_deform_mlp(deform_input)
        return deform
    
    def pose_deform(self, deform_input):
        deform = self.pose_deform_mlp(deform_input)
        return deform
        
    def get_landmarks(self):
        return self.landmarks_3d_neutral

    def subdivide(self):
        new_tet_verts, new_tets = kaolin.ops.mesh.subdivide_tetmesh(self.tet_verts.unsqueeze(0), self.tets)
        self.tet_verts = new_tet_verts[0]
        self.tets = new_tets
        self.grid_res *= 2

    def reconstruct(self, data):
        B = data['exp_coeff'].shape[0]

        query_pts = self.tet_verts.unsqueeze(0).repeat(B, 1, 1)
        geo_input = self.pos_embedding(query_pts).permute(0, 2, 1)

        pred = self.geometry(geo_input)

        sdf, deform, features = pred[:, :1, :], pred[:, 1:4, :], pred[:, 4:, :]
        sdf = sdf.permute(0, 2, 1)
        features = features.permute(0, 2, 1)
        verts_deformed = (query_pts + torch.tanh(deform.permute(0, 2, 1)) / self.grid_res)
        verts_list, features_list, faces_list = marching_tetrahedra(verts_deformed, features, self.tets, sdf)

        data['verts0_list'] = verts_list
        data['faces_list'] = faces_list

        verts_batch = []
        verts_features_batch = []
        num_pts_max = 0
        for b in range(B):
            if verts_list[b].shape[0] > num_pts_max:
                num_pts_max = verts_list[b].shape[0]
            
        for b in range(B):
            verts_batch.append(torch.cat([verts_list[b], torch.zeros([num_pts_max - verts_list[b].shape[0], verts_list[b].shape[1]], device=verts_list[b].device)], 0))
            verts_features_batch.append(torch.cat([features_list[b], torch.zeros([num_pts_max - features_list[b].shape[0], features_list[b].shape[1]], device=features_list[b].device)], 0))
        verts_batch = torch.stack(verts_batch, 0)
        verts_features_batch = torch.stack(verts_features_batch, 0)

        dists, idx, _ = knn_points(verts_batch, data['landmarks_3d_neutral'])
        exp_weights = torch.clamp((self.dist_threshold_far - dists) / (self.dist_threshold_far - self.dist_threshold_near), 0.0, 1.0)
        pose_weights = 1 - exp_weights

        exp_color_input = torch.cat([verts_features_batch.permute(0, 2, 1), data['exp_coeff'].unsqueeze(-1).repeat(1, 1, num_pts_max)], 1)
        verts_color_batch = self.exp_color(exp_color_input).permute(0, 2, 1) * exp_weights
        
        pose_color_input = torch.cat([verts_features_batch.permute(0, 2, 1), self.pos_embedding(data['pose']).unsqueeze(-1).repeat(1, 1, num_pts_max)], 1)
        verts_color_batch = verts_color_batch + self.pose_color(pose_color_input).permute(0, 2, 1) * pose_weights

        exp_deform_input = torch.cat([self.pos_embedding(verts_batch).permute(0, 2, 1), data['exp_coeff'].unsqueeze(-1).repeat(1, 1, num_pts_max)], 1)
        exp_deform = self.exp_deform(exp_deform_input).permute(0, 2, 1)
        verts_batch = verts_batch + exp_deform * exp_weights * self.deform_scale

        pose_deform_input = torch.cat([self.pos_embedding(verts_batch).permute(0, 2, 1), self.pos_embedding(data['pose']).unsqueeze(-1).repeat(1, 1, num_pts_max)], 1)
        pose_deform = self.pose_deform(pose_deform_input).permute(0, 2, 1)
        verts_batch = verts_batch + pose_deform * pose_weights * self.deform_scale

        if 'pose' in data:
            R = so3_exponential_map(data['pose'][:, :3])
            T = data['pose'][:, None, 3:]
            S = data['scale'][:, :, None]
            verts_batch = torch.bmm(verts_batch * S, R.permute(0, 2, 1)) + T

        data['exp_deform'] = exp_deform
        data['pose_deform'] = pose_deform
        data['verts_list'] = [verts_batch[b, :verts_list[b].shape[0], :] for b in range(B)]
        data['verts_color_list'] = [verts_color_batch[b, :verts_list[b].shape[0], :] for b in range(B)]
        return data
    
    def reconstruct_neutral(self):
        query_pts = self.tet_verts.unsqueeze(0)
        geo_input = self.pos_embedding(query_pts).permute(0, 2, 1)

        pred = self.geometry(geo_input)

        sdf, deform, features = pred[:, :1, :], pred[:, 1:4, :], pred[:, 4:, :]
        sdf = sdf.permute(0, 2, 1)
        features = features.permute(0, 2, 1)
        verts_deformed = (query_pts + torch.tanh(deform.permute(0, 2, 1)) / self.grid_res)
        verts_list, features_list, faces_list = marching_tetrahedra(verts_deformed, features, self.tets, sdf)

        data = {}
        data['verts'] = verts_list[0]
        data['faces'] = faces_list[0]
        data['verts_feature'] = features_list[0]
        return data
    
    def query_sdf(self, data):
        query_pts = data['query_pts']

        geo_input = self.pos_embedding(query_pts).permute(0, 2, 1)

        pred = self.geometry(geo_input)
        sdf = pred[:, :1, :]
        sdf = sdf.permute(0, 2, 1)

        data['sdf'] = sdf
        return data
    
    def deform(self, data):
        exp_coeff = data['exp_coeff']
        query_pts = data['query_pts']
        
        geo_input = self.pos_embedding(query_pts).permute(0, 2, 1)

        pred = self.geometry(geo_input)
        sdf, deform = pred[:, :1, :], pred[:, 1:4, :]
        query_pts = (query_pts + torch.tanh(deform).permute(0, 2, 1) / self.grid_res)

        exp_deform_input = torch.cat([self.pos_embedding(query_pts).permute(0, 2, 1), exp_coeff.unsqueeze(-1).repeat(1, 1, query_pts.shape[1])], 1)
        exp_deform = self.exp_deform(exp_deform_input).permute(0, 2, 1)

        deformed_pts = query_pts + exp_deform * self.deform_scale

        data['deformed_pts'] = deformed_pts
        return data
    
    def in_bbox(self, verts, bbox):
        is_in_bbox = (verts[:, :, 0] > bbox[0][0]) & \
                     (verts[:, :, 1] > bbox[1][0]) & \
                     (verts[:, :, 2] > bbox[2][0]) & \
                     (verts[:, :, 0] < bbox[0][1]) & \
                     (verts[:, :, 1] < bbox[1][1]) & \
                     (verts[:, :, 2] < bbox[2][1])
        return is_in_bbox
    
    def pre_train_sphere(self, iter, device):
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)

        for i in tqdm.tqdm(range(iter)):
            query_pts = torch.rand((8, 1024, 3), device=device) * 3 - 1.5
            ref_value  = torch.sqrt((query_pts**2).sum(-1)) - 1.0
            data = {
                'query_pts': query_pts
                }
            data = self.query_sdf(data)
            sdf = data['sdf']
            loss = loss_fn(sdf[:, :, 0], ref_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Pre-trained MLP", loss.item())