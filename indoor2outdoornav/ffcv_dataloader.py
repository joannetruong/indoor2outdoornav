import glob
import os.path as osp

import cv2
import numpy as np
from ffcv.fields import FloatField, NDArrayField
from ffcv.fields.decoders import FloatDecoder, NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor
from ffcv.writer import DatasetWriter


class OutdoorDataset:
    def __init__(self, dir_names):
        self.X = []
        for dir_name in dir_names:
            self.X.extend(glob.glob(osp.join(dir_name, "*.png")))

    def __getitem__(self, idx):
        filename = self.X[idx]
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        return img.astype("float32") / 255.0

    def __len__(self):
        return len(self.X)


def generate_dataset(dir_names):
    ds = OutdoorDataset(dir_names)
    write_path = "/coc/testnvme/jtruong33/data/outdoor_imgs/outdoor.beton"
    writer = DatasetWriter(
        write_path,
        {
            "img": NDArrayField(shape=(d,), dtype=np.dtype("float32")),
        },
        num_workers=16,
    )

    writer.from_indexed_dataset(ds)


def get_dataloader(write_path, batch_size=128, num_workers=16):
    return Loader(
        write_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.QUASI_RANDOM,
        pipelines={"img": [NDArrayDecoder(), ToTensor()]},
    )
