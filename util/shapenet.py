
import os
import glob
import torch
from torch.utils import data
import numpy as np


class ShapeNet(data.Dataset):

    def __init__(
        self,
        surfaces_folder,
        occupancies_folder,
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

        self.split = split
        self.num_query_pts = num_query_pts
        self.pc_size = pc_size
        self.return_surface = return_surface
        self.transform = transform

        self.surfaces_dir = os.path.join(
            surfaces_folder,
            category_id,
            '4_pointcloud'
        )

        self.occupancies_dir = os.path.join(
            occupancies_folder,
            'ShapeNetV2_point',
            category_id
        )

        occ_files = sorted(glob.glob(os.path.join(self.occupancies_dir,'*.npz')))

        matched_models = []

        for occ_file in occ_files:
            model_id = os.path.splitext(os.path.basename(occ_file))[0]
            surf_file = os.path.join(self.surfaces_dir,model_id + '.npz')

            if os.path.isfile(surf_file):
                matched_models.append((model_id, occ_file, surf_file))

        if len(matched_models) == 0:
            raise RuntimeError(
                "No matching occupancy/surface models found.\n"
                f"Occupancy directory: {self.occupancies_dir}\n"
                f"Surface directory: {self.surfaces_dir}"
            )

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(matched_models))

        matched_models = [matched_models[i]for i in indices]

        n_total = len(matched_models)

        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        train_models = matched_models[:n_train]
        val_models = matched_models[n_train: n_train + n_val]
        test_models = matched_models[n_train + n_val:]

        if split == 'train':
            self.models = train_models

        elif split == 'val':
            self.models = val_models

        elif split == 'test':
            self.models = test_models

        else:
            raise ValueError(
                f"Unknown split: {split}. "
                "Choose from ['train', 'val', 'test']"
            )

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):

        model_id, occ_file, surf_file = self.models[idx]

        with np.load(occ_file) as occ_data:

            query_points = occ_data['vol_points'].astype(np.float32)
            raw_labels = occ_data['vol_label']

        if raw_labels.dtype == np.uint8:

            labels = np.unpackbits(raw_labels)[:len(query_points)].astype(np.float32)
        else:
            labels = raw_labels.astype(np.float32)

        if (self.num_query_pts is not None and len(query_points) > self.num_query_pts):
            indices = np.random.choice(
                len(query_points),
                self.num_query_pts,
                replace=False
            )

            query_points = query_points[indices]

            labels = labels[indices]

        query_points = torch.from_numpy(query_points)

        labels = torch.from_numpy(labels)

        surface = None

        if self.return_surface:

            with np.load(surf_file) as surf_data:

                surface = surf_data['points'].astype(np.float32)

            if (self.pc_size is not None and len(surface) > self.pc_size):

                indices = np.random.choice(
                    len(surface),
                    self.pc_size,
                    replace=False
                )

                surface = surface[indices]

            surface = torch.from_numpy(surface)

        if self.transform is not None:
            if self.return_surface:
                surface, query_points = self.transform(surface,query_points)
            else:
                query_points = self.transform(query_points)

        if self.return_surface:
            return (query_points, labels, surface)
        return (query_points, labels)