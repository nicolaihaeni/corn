import os
import numpy as np
from numpy import loadtxt
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from models.transform.pose_utils import pose_from_filename
from utils.util import segment_image


class ShapenetDatasetSRN(Dataset):
    def __init__(self, opt, file_path):
        self.file_path = file_path
        self.num_views = 108
        self.opt = opt

        self.files = loadtxt(file_path, dtype=str, delimiter=',')
        self.length = len(self.files)

        # Angular increments
        self.n_azimuth_angles = 36
        self.azimuth_increment = 10
        self.n_elevations = 3
        self.elevation_increment = 10
        self.azimuth_increment = 10

        az_range = np.linspace(0,35,36)
        el_range = np.linspace(0,20,3)
        self.angles_list = np.array(np.meshgrid(az_range, el_range)).reshape(2,-1).astype(np.int).T

        # Source configuration
        self.src_azi = 34
        self.src_elev = 0

        self.downsample = transforms.Resize((128, 128))
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, idx):
        data = {}
        model_name = self.files[idx]
        angles_to_str = lambda x: f"{x[0]}_{x[1]}"

        src_path = os.path.join(self.opt.data_dir,
                                f'{model_name}_{self.src_azi}_{self.src_elev}.png')
        src_img = self.downsample(Image.open(src_path))
        src_img = np.array(src_img)

        data['model_name'] = model_name
        data['src_0'] = self.transforms(src_img)
        data['src_0_mask'] = torch.Tensor(segment_image(src_img))
        data[f'src_0_cam_pose'] = torch.Tensor(pose_from_filename(angles_to_str([self.src_azi, self.src_elev])))

        target_imgs = []
        target_masks = []
        target_poses = []
        angles_list = self.angles_list if not self.opt.random_views else np.random.permutation(len(self.angles_list))[:self.opt.num_views]

        for ii, jj in angles_list:
            path = os.path.join(self.opt.data_dir, f'{model_name}_{ii}_{jj*10}.png')
            img = self.downsample(Image.open(path))
            img = np.array(img)

            target_imgs.append(self.transforms(img))
            target_masks.append(torch.Tensor(segment_image(img)))
            target_poses.append(torch.Tensor(pose_from_filename(angles_to_str([ii, jj]))))

        data['target_imgs'] = target_imgs
        data['target_masks'] = target_masks
        data['target_poses'] = target_poses
        return data

    def __len__(self):
        return self.length
