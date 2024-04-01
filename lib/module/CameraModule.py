import torch
import kaolin
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


class CameraModule():
    def __init__(self):
        self.bg_color = torch.tensor([1.0] * 32).float()
        self.scale_modifier = 1.0

    def perspective_camera(self, points, camera_proj):
        projected_points = torch.bmm(points, camera_proj.permute(0, 2, 1))
        projected_2d_points = projected_points[:, :, :2] / projected_points[:, :, 2:3]

        return projected_2d_points

    def prepare_vertices(self, vertices, faces, camera_proj, camera_rot=None, camera_trans=None,
                     camera_transform=None):
        if camera_transform is None:
            assert camera_trans is not None and camera_rot is not None, \
                "camera_transform or camera_trans and camera_rot must be defined"
            vertices_camera = kaolin.render.camera.rotate_translate_points(vertices, camera_rot,
                                                            camera_trans)
        else:
            assert camera_trans is None and camera_rot is None, \
                "camera_trans and camera_rot must be None when camera_transform is defined"
            padded_vertices = torch.nn.functional.pad(
                vertices, (0, 1), mode='constant', value=1.
            )
            vertices_camera = (padded_vertices @ camera_transform)
        # Project the vertices on the camera image plan
        vertices_image = self.perspective_camera(vertices_camera, camera_proj)
        face_vertices_camera = kaolin.ops.mesh.index_vertices_by_faces(vertices_camera, faces)
        face_vertices_image = kaolin.ops.mesh.index_vertices_by_faces(vertices_image, faces)
        face_normals = kaolin.ops.mesh.face_normals(face_vertices_camera, unit=True)
        return face_vertices_camera, face_vertices_image, face_normals
    
    def render(self, data, resolution):
        verts_list = data['verts_list']
        faces_list = data['faces_list']
        verts_color_list = data['verts_color_list']

        B = len(verts_list)

        render_images = []
        render_soft_masks = []
        render_depths = []
        render_normals = []
        face_normals_list = []
        for b in range(B):
            intrinsics = data['intrinsics'][b]
            extrinsics = data['extrinsics'][b]
            #camera_proj = torch.stack([intrinsics[:, 0, 0] / intrinsics[:, 0, 2], intrinsics[:, 1, 1] / intrinsics[:, 1, 2], torch.ones_like(intrinsics[:, 0, 0])], -1).to(device)
            camera_proj = intrinsics
            camera_transform = extrinsics.permute(0, 2, 1)

            verts = verts_list[b].unsqueeze(0).repeat(intrinsics.shape[0], 1, 1)
            faces = faces_list[b]
            verts_color = verts_color_list[b].unsqueeze(0).repeat(intrinsics.shape[0], 1, 1)
            faces_color = verts_color[:, faces]

            face_vertices_camera, face_vertices_image, face_normals = self.prepare_vertices(
                verts, faces, camera_proj, camera_transform=camera_transform
            )
            face_vertices_image[:, :, :, 1] = -face_vertices_image[:, :, :, 1]
            #face_vertices_camera[:, :, :, 1:] = -face_vertices_camera[:, :, :, 1:]
            face_normals[:, :, 1:] = -face_normals[:, :, 1:]
            ### Perform Rasterization ###
            # Construct attributes that DI1-R rasterizer will interpolate.
            # the first is the UVS associated to each face
            # the second will make a hard segmentation mask
            face_attributes = [
                faces_color,
                torch.ones((faces_color.shape[0], faces_color.shape[1], 3, 1), device=verts.device),
                face_vertices_camera[:, :, :, 2:],
                face_normals.unsqueeze(-2).repeat(1, 1, 3, 1),
            ]

            # If you have nvdiffrast installed you can change rast_backend to
            # nvdiffrast or nvdiffrast_fwd
            image_features, soft_masks, face_idx = kaolin.render.mesh.dibr_rasterization(
                resolution, resolution, -face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1],
                rast_backend='cuda')

            # image_features is a tuple in composed of the interpolated attributes of face_attributes
            images, masks, depths, normals = image_features
            images = torch.clamp(images * masks, 0., 1.)
            depths = (depths * masks)
            normals = (normals * masks)
            
            render_images.append(images)
            render_soft_masks.append(soft_masks)
            render_depths.append(depths)
            render_normals.append(normals)
            face_normals_list.append(face_normals)

        render_images = torch.stack(render_images, 0)
        render_soft_masks = torch.stack(render_soft_masks, 0)
        render_depths = torch.stack(render_depths, 0)
        render_normals = torch.stack(render_normals, 0)

        data['render_images'] = render_images
        data['render_soft_masks'] = render_soft_masks
        data['render_depths'] = render_depths
        data['render_normals'] = render_normals
        data['verts_list'] = verts_list
        data['faces_list'] = faces_list
        data['face_normals_list'] = face_normals_list

        return data
    

    def render_gaussian(self, data, resolution):
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        """
        B = data['xyz'].shape[0]
        xyz = data['xyz']
        #shs = rearrange(data['shs'], 'b n (x y) -> b n x y', y=3)
        colors_precomp = data['color']
        opacity = data['opacity']
        scales = data['scales']
        rotations = data['rotation']
        fovx = data['fovx']
        fovy = data['fovy']

        world_view_transform = data['world_view_transform']
        full_proj_transform = data['full_proj_transform']
        camera_center = data['camera_center']
    
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device=xyz.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        render_images = []
        radii = []
        for b in range(B):

            tanfovx = math.tan(fovx[b] * 0.5)
            tanfovy = math.tan(fovy[b] * 0.5)
            # Set up rasterization configuration
            raster_settings = GaussianRasterizationSettings(
                image_height=int(resolution),
                image_width=int(resolution),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=self.bg_color.to(xyz.device),
                scale_modifier=self.scale_modifier,
                viewmatrix=world_view_transform[b],
                projmatrix=full_proj_transform[b],
                sh_degree=0,
                campos=camera_center[b],
                prefiltered=False,
                debug=False
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            means3D = xyz[b]
            means2D = screenspace_points[b]
            
            # Rasterize visible Gaussians to image, obtain their radii (on screen). 
            render_images_b, radii_b = rasterizer(
                means3D = means3D,
                means2D = means2D,
                #shs = shs[b],
                colors_precomp = colors_precomp[b],
                opacities = opacity[b],
                scales = scales[b],
                rotations = rotations[b])
            render_images.append(render_images_b)
            radii.append(radii_b)

        render_images = torch.stack(render_images)
        radii = torch.stack(radii)
        data['render_images'] = render_images
        data['viewspace_points'] =  screenspace_points
        data['visibility_filter'] = radii > 0
        data['radii'] = radii
        return data