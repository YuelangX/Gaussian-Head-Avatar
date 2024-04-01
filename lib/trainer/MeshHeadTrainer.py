import torch
import torch.nn.functional as F
from tqdm import tqdm
import kaolin
import lpips
from einops import rearrange
from pytorch3d.transforms import so3_exponential_map

from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance


def laplace_regularizer_const(mesh_verts, mesh_faces):
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)




class MeshHeadTrainer():
    def __init__(self, dataloader, meshhead, camera, optimizer, recorder, gpu_id):
        self.dataloader = dataloader
        self.meshhead = meshhead
        self.camera = camera
        self.optimizer = optimizer
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)

    def train(self, start_epoch=0, epochs=1):
        for epoch in range(start_epoch, epochs):
            for idx, data in tqdm(enumerate(self.dataloader)):
                
                # prepare data
                to_cuda = ['images', 'masks', 'visibles', 'intrinsics', 'extrinsics', 'pose', 'scale', 'exp_coeff', 'landmarks_3d', 'exp_id']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)

                images = data['images'].permute(0, 1, 3, 4, 2)
                masks = data['masks'].permute(0, 1, 3, 4, 2)
                visibles = data['visibles'].permute(0, 1, 3, 4, 2)
                resolution = images.shape[2]

                R = so3_exponential_map(data['pose'][:, :3])
                T = data['pose'][:, 3:, None]
                S = data['scale'][:, :, None]
                landmarks_3d_can = (torch.bmm(R.permute(0,2,1), (data['landmarks_3d'].permute(0, 2, 1) - T)) / S).permute(0, 2, 1)
                landmarks_3d_neutral = self.meshhead.get_landmarks()[None].repeat(data['landmarks_3d'].shape[0], 1, 1)
                data['landmarks_3d_neutral'] = landmarks_3d_neutral

                deform_data = {
                    'exp_coeff': data['exp_coeff'],
                    'query_pts': landmarks_3d_neutral
                }
                deform_data = self.meshhead.deform(deform_data)
                pred_landmarks_3d_can = deform_data['deformed_pts']
                loss_def = F.mse_loss(pred_landmarks_3d_can, landmarks_3d_can)

                deform_data = self.meshhead.query_sdf(deform_data)
                sdf_landmarks_3d = deform_data['sdf']
                loss_lmk = torch.abs(sdf_landmarks_3d[:, :, 0]).mean()

                data = self.meshhead.reconstruct(data)
                data = self.camera.render(data, resolution)
                render_images = data['render_images']
                render_soft_masks = data['render_soft_masks']
                exp_deform = data['exp_deform']
                pose_deform = data['pose_deform']
                verts_list = data['verts_list']
                faces_list = data['faces_list']

                loss_rgb = F.l1_loss(render_images[:, :, :, :, 0:3] * visibles, images * visibles)
                loss_sil = kaolin.metrics.render.mask_iou((render_soft_masks * visibles[:, :, :, :, 0]).view(-1, resolution, resolution), (masks * visibles).squeeze().view(-1, resolution, resolution))
                loss_offset = (exp_deform ** 2).sum(-1).mean() + (pose_deform ** 2).sum(-1).mean()

                loss_lap = 0.0
                for b in range(len(verts_list)):
                    loss_lap += laplace_regularizer_const(verts_list[b], faces_list[b])
                
                loss = loss_rgb * 1e-1 + loss_sil * 1e-1 + loss_def * 1e0 + loss_offset * 1e-2 + loss_lmk * 1e-1 + loss_lap * 1e2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                log = {
                    'data': data,
                    'meshhead' : self.meshhead,
                    'loss_rgb' : loss_rgb,
                    'loss_sil' : loss_sil,
                    'loss_def' : loss_def,
                    'loss_offset' : loss_offset,
                    'loss_lmk' : loss_lmk,
                    'loss_lap' : loss_lap,
                    'epoch' : epoch,
                    'iter' : idx + epoch * len(self.dataloader)
                }
                self.recorder.log(log)
