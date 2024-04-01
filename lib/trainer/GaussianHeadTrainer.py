import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import lpips


class GaussianHeadTrainer():
    def __init__(self, dataloader, delta_poses, gaussianhead, supres, camera, optimizer, recorder, gpu_id):
        self.dataloader = dataloader
        self.delta_poses = delta_poses
        self.gaussianhead = gaussianhead
        self.supres = supres
        self.camera = camera
        self.optimizer = optimizer
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)
        self.fn_lpips = lpips.LPIPS(net='vgg').to(self.device)

    def train(self, start_epoch=0, epochs=1):
        for epoch in range(start_epoch, epochs):
            for idx, data in tqdm(enumerate(self.dataloader)):
                
                # prepare data
                to_cuda = ['images', 'masks', 'visibles', 'images_coarse', 'masks_coarse', 'visibles_coarse', 
                           'intrinsics', 'extrinsics', 'world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center',
                           'pose', 'scale', 'exp_coeff', 'landmarks_3d', 'exp_id']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)

                images = data['images']
                visibles = data['visibles']
                if self.supres is None:
                    images_coarse = images
                    visibles_coarse = visibles
                else:
                    images_coarse = data['images_coarse']
                    visibles_coarse = data['visibles_coarse']

                resolution_coarse = images_coarse.shape[2]
                resolution_fine = images.shape[2]

                data['pose'] = data['pose'] + self.delta_poses[data['exp_id'], :]

                # render coarse images
                data = self.gaussianhead.generate(data)
                data = self.camera.render_gaussian(data, resolution_coarse)
                render_images = data['render_images']

                # crop images for augmentation
                scale_factor = random.random() * 0.45 + 0.8
                scale_factor = int(resolution_coarse * scale_factor) / resolution_coarse
                cropped_render_images, cropped_images, cropped_visibles = self.random_crop(render_images, images, visibles, scale_factor, resolution_coarse, resolution_fine)
                data['cropped_images'] = cropped_images
                
                # generate super resolution images
                supres_images = self.supres(cropped_render_images)
                data['supres_images'] = supres_images

                # loss functions
                loss_rgb_lr = F.l1_loss(render_images[:, 0:3, :, :] * visibles_coarse, images_coarse * visibles_coarse)
                loss_rgb_hr = F.l1_loss(supres_images * cropped_visibles, cropped_images * cropped_visibles)
                left_up = (random.randint(0, supres_images.shape[2] - 512), random.randint(0, supres_images.shape[3] - 512))
                loss_vgg = self.fn_lpips((supres_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], 
                                            (cropped_images * cropped_visibles)[:, :, left_up[0]:left_up[0]+512, left_up[1]:left_up[1]+512], normalize=True).mean()
                loss = loss_rgb_hr + loss_rgb_lr + loss_vgg * 1e-1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                log = {
                    'data': data,
                    'delta_poses' : self.delta_poses,
                    'gaussianhead' : self.gaussianhead,
                    'supres' : self.supres,
                    'loss_rgb_lr' : loss_rgb_lr,
                    'loss_rgb_hr' : loss_rgb_hr,
                    'loss_vgg' : loss_vgg,
                    'epoch' : epoch,
                    'iter' : idx + epoch * len(self.dataloader)
                }
                self.recorder.log(log)


    def random_crop(self, render_images, images, visibles, scale_factor, resolution_coarse, resolution_fine):
        render_images_scaled = F.interpolate(render_images, scale_factor=scale_factor)
        images_scaled = F.interpolate(images, scale_factor=scale_factor)
        visibles_scaled = F.interpolate(visibles, scale_factor=scale_factor)

        if scale_factor < 1:
            render_images = torch.ones([render_images_scaled.shape[0], render_images_scaled.shape[1], resolution_coarse, resolution_coarse], device=self.device)
            left_up_coarse = (random.randint(0, resolution_coarse - render_images_scaled.shape[2]), random.randint(0, resolution_coarse - render_images_scaled.shape[3]))
            render_images[:, :, left_up_coarse[0]: left_up_coarse[0] + render_images_scaled.shape[2], left_up_coarse[1]: left_up_coarse[1] + render_images_scaled.shape[3]] = render_images_scaled

            images = torch.ones([images_scaled.shape[0], images_scaled.shape[1], resolution_fine, resolution_fine], device=self.device)
            visibles = torch.ones([visibles_scaled.shape[0], visibles_scaled.shape[1], resolution_fine, resolution_fine], device=self.device)
            left_up_fine = (int(left_up_coarse[0] * resolution_fine / resolution_coarse), int(left_up_coarse[1] * resolution_fine / resolution_coarse))
            images[:, :, left_up_fine[0]: left_up_fine[0] + images_scaled.shape[2], left_up_fine[1]: left_up_fine[1] + images_scaled.shape[3]] = images_scaled
            visibles[:, :, left_up_fine[0]: left_up_fine[0] + visibles_scaled.shape[2], left_up_fine[1]: left_up_fine[1] + visibles_scaled.shape[3]] = visibles_scaled
        else:
            left_up_coarse = (random.randint(0, render_images_scaled.shape[2] - resolution_coarse), random.randint(0, render_images_scaled.shape[3] - resolution_coarse))
            render_images = render_images_scaled[:, :, left_up_coarse[0]: left_up_coarse[0] + resolution_coarse, left_up_coarse[1]: left_up_coarse[1] + resolution_coarse]

            left_up_fine = (int(left_up_coarse[0] * resolution_fine / resolution_coarse), int(left_up_coarse[1] * resolution_fine / resolution_coarse))
            images = images_scaled[:, :, left_up_fine[0]: left_up_fine[0] + resolution_fine, left_up_fine[1]: left_up_fine[1] + resolution_fine]
            visibles = visibles_scaled[:, :, left_up_fine[0]: left_up_fine[0] + resolution_fine, left_up_fine[1]: left_up_fine[1] + resolution_fine]
        
        return render_images, images, visibles