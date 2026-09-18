
import os
import glob
import torch
from torch.utils import data

import numpy as np

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

        occ_models = glob.glob(os.path.join(self.occupancies_dir, '*.npz'))
        sur_models = [os.path.join(self.surfaces_dir, os.path.splitext(os.path.basename(m))[0], '.npz')
                       for m in occ_models
                       if os.path.exists(
                        os.path.join(self.surfaces_dir, os.path.splitext(os.path.basename(m))[0] + '.npz')
                        )]

        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(len(occ_models))
        
        n_train = int(train_ratio * len(occ_models))
        n_val = int(val_ratio * len(occ_models))

        if split == 'train':
            selected_idx = shuffled_indices[:n_train]
        elif split == 'val':
            selected_idx = shuffled_indices[n_train:n_train + n_val]
        elif split == 'test':
            selected_idx = shuffled_indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}. Choose from ['train', 'val', 'test']")

        self.models = [occ_models[i] for i in selected_idx]

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model_id = self.models[idx]

        occ_file = os.path.join(self.occupancies_dir, model_id)
        with np.load(occ_file) as occ_data:
            query_points = occ_data['vol_points'].astype(np.float32)
            raw_labels = occ_data['vol_label']

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