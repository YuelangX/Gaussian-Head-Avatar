import torch
import torch.nn.functional as F
from tqdm import tqdm


class Reenactment():
    def __init__(self, dataloader, gaussianhead, supres, camera, recorder, gpu_id, freeview):
        self.dataloader = dataloader
        self.gaussianhead = gaussianhead
        self.supres = supres
        self.camera = camera
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)
        self.freeview = freeview

    def run(self):
        for idx, data in tqdm(enumerate(self.dataloader)):

            to_cuda = ['images', 'intrinsics', 'extrinsics', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center', 
                       'pose', 'scale', 'exp_coeff', 'pose_code']
            for data_item in to_cuda:
                data[data_item] = data[data_item].to(device=self.device)

            if not self.freeview:
                if idx > 0:
                    data['pose'] = pose_last * 0.5 + data['pose'] * 0.5
                    data['exp_coeff'] = exp_last * 0.5 + data['exp_coeff'] * 0.5
                pose_last = data['pose']
                exp_last = data['exp_coeff']
                
            else:
                data['pose'] *= 0
                if idx > 0:
                    data['exp_coeff'] = exp_last * 0.5 + data['exp_coeff'] * 0.5
                exp_last = data['exp_coeff']
            
            with torch.no_grad():
                data = self.gaussianhead.generate(data)
                data = self.camera.render_gaussian(data, 512)
                render_images = data['render_images']
                supres_images = self.supres(render_images)
                data['supres_images'] = supres_images

            log = {
                'data': data,
                'iter': idx
            }
            self.recorder.log(log)
