
import os
import glob
import torch
from torch.utils import data
import numpy as np


class ShapeNet(data.Dataset):

    def __init__(
        self,
        mesh_folder,
        point_folder,
        split='train',
        category_id='03001627',
        num_samples=4096,
        pc_size=2048,
        return_surface=True,
        surface_sampling=True,
        sampling=True,
        transform=None,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
        replica=16
    ):

        self.split = split
        self.num_samples = num_samples
        self.pc_size = pc_size
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling
        self.sampling = sampling
        self.transform = transform
        self.replica=replica

        self.mesh_folder = os.path.join(mesh_folder, category_id, '4_pointcloud')
        self.point_folder = os.path.join(point_folder, 'ShapeNetV2_point', category_id)
        
        self.models = glob.glob(os.path.join(self.point_folder, '*.npz'))
        rng = np.random.default_rng(seed)
        rng.shuffle(self.models)

        n_total = len(self.models)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        
        if split == 'train':
            self.models = self.models[:n_train]

        elif split == 'val':
            self.models = self.models[n_train: n_train + n_val]

        elif split == 'test':
            self.models = self.models[n_train + n_val:]

        else:
            raise ValueError(f"Unknown split: {split}.")

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return len(self.models) * self.replica

    def __getitem__(self, idx):
        idx = idx % len(self.models)

        model_path = self.models[idx]
        try:
            with np.load(model_path) as data:
                vol_points = data['vol_points']
                vol_label = data['vol_label']
                near_points = data['near_points']
                near_label = data['near_label']
        except Exception as e:
            print(e)
            print(model_path)

        with open(model_path.replace('.npz', '.npy'), 'rb') as f:
            scale = np.load(f).item()

        if self.return_surface:
            model_idx = os.path.basename(model_path)
            pc_path = os.path.join(self.mesh_folder, model_idx)
            with np.load(pc_path) as data:
                surface = data['points'].astype(np.float32)
                surface = surface * scale
            if self.surface_sampling:
                ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                surface = surface[ind]
            surface = torch.from_numpy(surface)

        if self.sampling:
            ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
            near_points = near_points[ind]
            near_label = near_label[ind]

        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label).float()

        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_label, near_label], dim=0)
        else:
            points = vol_points
            labels = vol_label

        if self.transform:
            surface, points = self.transform(surface, points)

        if self.return_surface:
            return points, labels, surface
        else:
            return points, labels