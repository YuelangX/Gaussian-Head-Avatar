import os
import torch
import argparse

from config.config import config_train

from lib.dataset.Dataset import GaussianDataset
from lib.dataset.DataLoaderX import DataLoaderX
from lib.module.MeshHeadModule import MeshHeadModule
from lib.module.GaussianHeadModule import GaussianHeadModule
from lib.module.SuperResolutionModule import SuperResolutionModule
from lib.module.CameraModule import CameraModule
from lib.recorder.Recorder import GaussianHeadTrainRecorder
from lib.trainer.GaussianHeadTrainer import GaussianHeadTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_s2_N031.yaml')
    arg = parser.parse_args()

    cfg = config_train()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    dataset = GaussianDataset(cfg.dataset)
    dataloader = DataLoaderX(dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True) 

    device = torch.device('cuda:%d' % cfg.gpu_id)
    
    if os.path.exists(cfg.load_gaussianhead_checkpoint):
        gaussianhead_state_dict = torch.load(cfg.load_gaussianhead_checkpoint, map_location=lambda storage, loc: storage)
        gaussianhead = GaussianHeadModule(cfg.gaussianheadmodule, 
                                          xyz=gaussianhead_state_dict['xyz'], 
                                          feature=gaussianhead_state_dict['feature'],
                                          landmarks_3d_neutral=gaussianhead_state_dict['landmarks_3d_neutral']).to(device)
        gaussianhead.load_state_dict(gaussianhead_state_dict)
    else:
        meshhead_state_dict = torch.load(cfg.load_meshhead_checkpoint, map_location=lambda storage, loc: storage)
        meshhead = MeshHeadModule(cfg.meshheadmodule, meshhead_state_dict['landmarks_3d_neutral']).to(device)
        meshhead.load_state_dict(meshhead_state_dict)
        meshhead.subdivide()
        with torch.no_grad():
            data = meshhead.reconstruct_neutral()

        gaussianhead = GaussianHeadModule(cfg.gaussianheadmodule, 
                                          xyz=data['verts'].cpu(),
                                          feature=torch.atanh(data['verts_feature'].cpu()), 
                                          landmarks_3d_neutral=meshhead.landmarks_3d_neutral.detach().cpu(),
                                          add_mouth_points=True).to(device)
        gaussianhead.exp_color_mlp.load_state_dict(meshhead.exp_color_mlp.state_dict())
        gaussianhead.pose_color_mlp.load_state_dict(meshhead.pose_color_mlp.state_dict())
        gaussianhead.exp_deform_mlp.load_state_dict(meshhead.exp_deform_mlp.state_dict())
        gaussianhead.pose_deform_mlp.load_state_dict(meshhead.pose_deform_mlp.state_dict())
    
    supres = SuperResolutionModule(cfg.supresmodule).to(device)
    if os.path.exists(cfg.load_supres_checkpoint):
        supres.load_state_dict(torch.load(cfg.load_supres_checkpoint, map_location=lambda storage, loc: storage))

    camera = CameraModule()
    recorder = GaussianHeadTrainRecorder(cfg.recorder)

    optimized_parameters = [{'params' : supres.parameters(), 'lr' : cfg.lr_net},
                            {'params' : gaussianhead.xyz, 'lr' : cfg.lr_net * 0.1},
                            {'params' : gaussianhead.feature, 'lr' : cfg.lr_net * 0.1},
                            {'params' : gaussianhead.exp_color_mlp.parameters(), 'lr' : cfg.lr_net},
                            {'params' : gaussianhead.pose_color_mlp.parameters(), 'lr' : cfg.lr_net},
                            {'params' : gaussianhead.exp_deform_mlp.parameters(), 'lr' : cfg.lr_net},
                            {'params' : gaussianhead.pose_deform_mlp.parameters(), 'lr' : cfg.lr_net},
                            {'params' : gaussianhead.exp_attributes_mlp.parameters(), 'lr' : cfg.lr_net},
                            {'params' : gaussianhead.pose_attributes_mlp.parameters(), 'lr' : cfg.lr_net},
                            {'params' : gaussianhead.scales, 'lr' : cfg.lr_net * 0.3},
                            {'params' : gaussianhead.rotation, 'lr' : cfg.lr_net * 0.1},
                            {'params' : gaussianhead.opacity, 'lr' : cfg.lr_net}]
    
    if os.path.exists(cfg.load_delta_poses_checkpoint):
        delta_poses = torch.load(cfg.load_delta_poses_checkpoint)
    else:
        delta_poses = torch.zeros([dataset.num_exp_id, 6]).to(device)

    if cfg.optimize_pose:
        delta_poses = delta_poses.requires_grad_(True)
        optimized_parameters.append({'params' : delta_poses, 'lr' : cfg.lr_pose})
    else:
        delta_poses = delta_poses.requires_grad_(False)

    optimizer = torch.optim.Adam(optimized_parameters)

    trainer = GaussianHeadTrainer(dataloader, delta_poses, gaussianhead, supres, camera, optimizer, recorder, cfg.gpu_id)
    trainer.train(0, 1000)

