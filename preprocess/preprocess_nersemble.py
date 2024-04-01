import os
import numpy as np
import cv2
import glob
import json


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




def extract_frames(id_list):

    for id in id_list:
        camera_path = os.path.join(DATA_SOURCE, 'camera_params', id, 'camera_params.json')
        with open(camera_path, 'r') as f:
            camera = json.load(f)

        fids = {}
        for camera_id in camera['world_2_cam'].keys():
            fids[camera_id] = 0
            background_path = os.path.join(DATA_SOURCE, 'sequence_BACKGROUND_part-1', id, 'BACKGROUND', 'image_%s.jpg' % camera_id)
            background = cv2.imread(background_path)
            background, _ = CropImage(LEFT_UP, CROP_SIZE, background, None)
            background, _ = ResizeImage(SIZE, CROP_SIZE, background, None)
            os.makedirs(os.path.join(DATA_OUTPUT, id, 'background'), exist_ok=True)
            cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'background', 'image_' + camera_id + '.jpg'), background)
        
        video_folders = glob.glob(os.path.join(DATA_SOURCE, '*', id, '*'))
        for video_folder in video_folders:
            if ('tongue' in video_folder) or ('GLASSES' in video_folder) or ('FREE' in video_folder) or ('BACKGROUND' in video_folder):
                continue
            video_paths = glob.glob(os.path.join(video_folder, 'cam_*'))
            for video_path in video_paths:
                camera_id = video_path[-13:-4]
                extrinsic = np.array(camera['world_2_cam'][camera_id][:3])
                intrinsic = np.array(camera['intrinsics'])
                _, intrinsic = CropImage(LEFT_UP, CROP_SIZE, None, intrinsic)
                _, intrinsic = ResizeImage(SIZE, CROP_SIZE, None, intrinsic)
                
                cap = cv2.VideoCapture(video_path)
                count = -1
                while(1): 
                    _, image = cap.read()
                    if image is None:
                        break
                    count += 1
                    if count % 3 != 0:
                        continue
                    visible = (np.ones_like(image) * 255).astype(np.uint8)
                    image, _ = CropImage(LEFT_UP, CROP_SIZE, image, None)
                    image, _ = ResizeImage(SIZE, CROP_SIZE, image, None)
                    visible, _ = CropImage(LEFT_UP, CROP_SIZE, visible, None)
                    visible, _ = ResizeImage(SIZE, CROP_SIZE, visible, None)
                    image_lowres = cv2.resize(image, SIZE_LOWRES)
                    visible_lowres = cv2.resize(visible, SIZE_LOWRES)
                    os.makedirs(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id]), exist_ok=True)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'image_' + camera_id + '.jpg'), image)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'image_lowres_' + camera_id + '.jpg'), image_lowres)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'visible_' + camera_id + '.jpg'), visible)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'visible_lowres_' + camera_id + '.jpg'), visible_lowres)
                    os.makedirs(os.path.join(DATA_OUTPUT, id, 'cameras', '%04d' % fids[camera_id]), exist_ok=True)
                    np.savez(os.path.join(DATA_OUTPUT, id, 'cameras', '%04d' % fids[camera_id], 'camera_' + camera_id + '.npz'), extrinsic=extrinsic, intrinsic=intrinsic)
                    
                    fids[camera_id] += 1
                    

if __name__ == "__main__":
    LEFT_UP = [-200, 304]
    CROP_SIZE = [2600, 2600]
    SIZE = [2048, 2048]
    SIZE_LOWRES = [256, 256]
    DATA_SOURCE = 'path/to/raw_NeRSemble/'
    DATA_OUTPUT = '../NeRSemble'
    extract_frames(['031', '036'])