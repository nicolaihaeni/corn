import random
import sys
import h5py
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from models.transform.pose_utils import pose_from_filename
from utils.util import segment_image


class ShapenetDataset(Dataset):
    def __init__(self, opt, file_path, is_train=True):
        self.file_path = file_path
        self.h5_file = None
        self.is_train = is_train
        self.num_views = opt.num_views
        self.opt = opt

        # Extract lenght
        with h5py.File(self.file_path, 'r') as h5_file:
            model_names = h5_file['ModelNames']
            model_names, ind = np.unique(model_names, return_index=True)
            ind = np.argsort(ind)
            model_names = model_names[ind]
            self.length = len(h5_file['ModelNames'])

        # Angular increments
        self.n_azimuth_angles = 36
        self.azimuth_increment = 10
        self.n_elevations = 3
        self.elevation_increment = 10
        self.azimuth_increment = 10

        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Fix two random indices per object model for training
        randints = []
        for ii in range(len(model_names)):
            randints.append(np.random.permutation(108)[:self.num_views])
        self.model_name_randint = dict(zip(model_names, randints))

    def __getitem__(self, idx):
        # Open the file once per thread
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, 'r')

        data = {}
        if self.is_train:
            model_name = self.h5_file['ModelNames'][idx][0]
            azi = self.h5_file['Azimuths'][idx].item()
            elev = self.h5_file['Elevations'][idx].item()
            base_idx = int(idx - (azi * 3) - (elev / 10))

            angles_to_str = lambda x: f"{x[0]}_{x[1]}"
            for ii in range(self.opt.num_views):
                idx = base_idx + self.model_name_randint[model_name][ii]
                img = self.h5_file['Images'][idx].astype(np.uint8)
                azi = self.h5_file['Azimuths'][idx].item()
                elev = self.h5_file['Elevations'][idx].item()
                data[f'src_{ii}'] = self.transforms(img)
                data[f'src_{ii}_mask'] = torch.Tensor(segment_image(img)).unsqueeze(0)
                data[f'src_{ii}_cam_pose'] = torch.Tensor(pose_from_filename(angles_to_str([azi, elev])))

            # Retrive a random target image
            a = random.randint(0, self.n_azimuth_angles - 1)
            e = random.randint(0, self.n_elevations - 1)
            idx = base_idx + a * 3 + e
            azi = self.h5_file['Azimuths'][idx].item()
            elev = self.h5_file['Elevations'][idx].item()
            img = self.h5_file['Images'][idx].astype(np.uint8)
            data[f'tgt'] = self.transforms(img)
            data[f'tgt_mask'] = torch.Tensor(segment_image(img)).unsqueeze(0)
            data[f'tgt_cam_pose'] = torch.Tensor(pose_from_filename(angles_to_str([azi, elev])))
            data['model_name'] = model_name.decode()
        else:
            images = self.h5_file['Images'][idx].astype(np.uint8)
            source_azimuth, target_azimuth = self.h5_file['Azimuths'][idx]
            source_elevation, target_elevation = self.h5_file['Elevations'][idx]
            model_name = self.h5_file['ModelNames'][idx][0].decode()
            source_mask = segment_image(images[0])
            target_mask = segment_image(images[1])

            # Compute camera pose
            angles_to_str = lambda x: f"{x[0]}_{x[1]}"
            src_cam_pose = torch.Tensor(pose_from_filename(angles_to_str([source_azimuth.item(), source_elevation.item()])))
            tgt_cam_pose = torch.Tensor(pose_from_filename(angles_to_str([target_azimuth.item(), target_elevation.item()])))

            data = {'src_0': self.transforms(images[0]),
                    'tgt': self.transforms(images[1]),
                    'src_0_mask': torch.Tensor(source_mask).unsqueeze(0),
                    'tgt_mask': torch.Tensor(target_mask).unsqueeze(0),
                    'src_0_cam_pose': src_cam_pose,
                    'tgt_cam_pose': tgt_cam_pose,
                    'model_name': model_name}
        return data

    def __len__(self):
        return self.length
