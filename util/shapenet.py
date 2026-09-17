
import os
import glob
import random

import yaml 

import torch
from torch.utils import data

import numpy as np

from PIL import Image

import h5py

category_ids = {
    '02691156': 0,
    '02747177': 1,
    '02773838': 2,
    '02801938': 3,
    '02808440': 4,
    '02818832': 5,
    '02828884': 6,
    '02843684': 7,
    '02871439': 8,
    '02876657': 9, 
    '02880940': 10,
    '02924116': 11,
    '02933112': 12,
    '02942699': 13,
    '02946921': 14,
    '02954340': 15,
    '02958343': 16,
    '02992529': 17,
    '03001627': 18,
    '03046257': 19,
    '03085013': 20,
    '03207941': 21,
    '03211117': 22,
    '03261776': 23,
    '03325088': 24,
    '03337140': 25,
    '03467517': 26,
    '03513137': 27,
    '03593526': 28,
    '03624134': 29,
    '03636649': 30,
    '03642806': 31,
    '03691459': 32,
    '03710193': 33,
    '03759954': 34,
    '03761084': 35,
    '03790512': 36,
    '03797390': 37,
    '03928116': 38,
    '03938244': 39,
    '03948459': 40,
    '03991062': 41,
    '04004475': 42,
    '04074963': 43,
    '04090263': 44,
    '04099429': 45,
    '04225987': 46,
    '04256520': 47,
    '04330267': 48,
    '04379243': 49,
    '04401088': 50,
    '04460130': 51,
    '04468005': 52,
    '04530566': 53,
    '04554684': 54,
}

class ShapeNet(data.Dataset):
    def __init__(
        self, 
        surfaces_folder=None, 
        occupancies_folder=None, 
        split='train', 
        category_id='03001627',
        num_query_pts=2048, 
        pc_size=2048, 
        return_surface=True, 
        transform=None, 
        train_ratio=0.8, 
        val_ratio=0.1,
        seed=42
    ):
        
        self.surfaces_dir = os.path.join(surfaces_folder, category_id, '4_pointcloud')
        self.occupancies_dir = os.path.join(occupancies_folder, 'ShapeNetV2_point', category_id)
        
        self.split = split
        self.num_query_pts = num_query_pts
        self.pc_size = pc_size
        self.return_surface = return_surface
        self.transform = transform

        surf_models = set(glob.glob(os.path.join(self.surfaces_dir, '*.npz')))
        occ_models = set(glob.glob(os.path.join(self.occupancies_dir)))
        common_models = sorted(list(surf_models.intersection(occ_models)))

        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(len(common_models))
        
        n_train = int(train_ratio * len(common_models))
        n_val = int(val_ratio * len(common_models))

        if split == 'train':
            selected_idx = shuffled_indices[:n_train]
        elif split == 'val':
            selected_idx = shuffled_indices[n_train:n_train + n_val]
        elif split == 'test':
            selected_idx = shuffled_indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}. Choose from ['train', 'val', 'test']")

        self.models = [common_models[i] for i in selected_idx]

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model_id = self.models[idx]

        occ_file = os.path.join(self.occupancies_dir, model_id)
        with np.load(occ_file) as occ_data:
            query_points = occ_data['points'].astype(np.float32)
            raw_labels = occ_data['occupancies']

        if raw_labels.dtype == np.uint8:
            labels = np.unpackbits(raw_labels)[:query_points.shape[0]].astype(np.float32)
        else:
            labels = raw_labels.astype(np.float32)

        if self.num_query_pts and len(query_points) >= self.num_query_pts:
            ind_occ = np.random.choice(len(query_points), self.num_query_pts, replace=False)
            query_points = query_points[ind_occ]
            labels = labels[ind_occ]

        query_points = torch.from_numpy(query_points)
        labels = torch.from_numpy(labels)

        surface = None
        if self.return_surface:
            surf_file = os.path.join(self.surfaces_dir, model_id)
            with np.load(surf_file) as surf_data:
                surf_pts = surf_data['points'].astype(np.float32)

            if self.pc_size and len(surf_pts) >= self.pc_size:
                ind_surf = np.random.choice(len(surf_pts), self.pc_size, replace=False)
                surf_pts = surf_pts[ind_surf]

            surface = torch.from_numpy(surf_pts)

        if self.transform:
            if self.return_surface:
                surface, query_points = self.transform(surface, query_points)
            else:
                query_points = self.transform(query_points)

        if self.return_surface:
            return query_points, labels, surface
        return query_points, labels