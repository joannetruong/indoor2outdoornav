import glob
import os.path as osp

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class OutdoorDataset(Dataset):
    def __init__(self, dir_names):
        super(Dataset, self).__init__()

        self.X = []
        for dir_name in dir_names:
            self.X.extend(glob.glob(osp.join(dir_name, "*.png")))

    def __getitem__(self, idx):
        filename = self.X[idx]
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=-1)
        return img.astype("float32") / 255.0

    def __len__(self):
        return len(self.X)


def get_dataloader(dir_names, batch_size=128, num_workers=8):
    return DataLoader(
        OutdoorDataset(dir_names),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=False,
    )
