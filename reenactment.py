import os
import torch
import argparse

from config.config import config_reenactment

from lib.dataset.Dataset import ReenactmentDataset
from lib.dataset.DataLoaderX import DataLoaderX
from lib.module.GaussianHeadModule import GaussianHeadModule
from lib.module.SuperResolutionModule import SuperResolutionModule
from lib.module.CameraModule import CameraModule
from lib.recorder.Recorder import ReenactmentRecorder
from lib.apps.Reenactment import Reenactment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/reenactment_N031.yaml')
    arg = parser.parse_args()

    cfg = config_reenactment()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    dataset = ReenactmentDataset(cfg.dataset)
    dataloader = DataLoaderX(dataset, batch_size=1, shuffle=False, pin_memory=True) 

    device = torch.device('cuda:%d' % cfg.gpu_id)

    gaussianhead_state_dict = torch.load(cfg.load_gaussianhead_checkpoint, map_location=lambda storage, loc: storage)
    gaussianhead = GaussianHeadModule(cfg.gaussianheadmodule, 
                                          xyz=gaussianhead_state_dict['xyz'], 
                                          feature=gaussianhead_state_dict['feature'],
                                          landmarks_3d_neutral=gaussianhead_state_dict['landmarks_3d_neutral']).to(device)
    gaussianhead.load_state_dict(gaussianhead_state_dict)

    supres = SuperResolutionModule(cfg.supresmodule).to(device)
    supres.load_state_dict(torch.load(cfg.load_supres_checkpoint, map_location=lambda storage, loc: storage))

    camera = CameraModule()
    recorder = ReenactmentRecorder(cfg.recorder)

    app = Reenactment(dataloader, gaussianhead, supres, camera, recorder, cfg.gpu_id, dataset.freeview)
    app.run()
