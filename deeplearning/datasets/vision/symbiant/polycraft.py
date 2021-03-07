import os

import numpy as np
import pandas as pd
import torch
from deeplearning.datasets.base import Dataset
from deeplearning.datasets.vision.base import _VisionDataset
from numpy.lib.twodim_base import mask_indices
from PIL import Image
from torchvision.transforms import transforms as _transforms


class PolycraftMappingDataset(_VisionDataset):

    mapping = {
        'H': 0,
        '.': 0,
        '_': 1,
        'W': 2,
        'D': 3,
        'T': 4,
        'G': 5,
    }

    reverse_mapping = {
        0: '.',
        1: '_',
        2: 'W',
        3: 'D',
        4: 'T',
        5: 'G'
    }

    color_mapping = {
        0: (0, 0, 0),
        1: (255, 153, 0),
        2: (179, 179, 179),
        3: (255, 194, 102),
        4: (0, 153, 0),
        5: (0, 0, 255)
    }
    class_weights = [1.0, 2.126, 8.737, 43.547, 1645.44 / 10., 1656.092 / 10.]

    __normalization__ = ((.3908, .3901, .4427), (.1761, .2316, .3159))

    def __init__(self, path, root_path, val_percent):
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.target_transforms = target_transforms

        if self.dataset_type is not Dataset.DatasetType.predict:

            data = pd.read_csv(path)
            data['config'] = data['image'].apply(
                lambda x: os.path.basename(os.path.dirname(x)))
            data['image'] = root_path + data['image']
            data['truth'] = root_path + data['truth']

            data_groups = data.groupby('config')
            data = data.drop('config', axis='columns')

            group_indicies = np.arange(data_groups.ngroups)
            np.random.seed(42)
            np.random.shuffle(group_indicies)

            train_indicies = data_groups.ngroup().isin(
                group_indicies[int(len(group_indicies) * val_frac):])

            self.data = data

        else:
            data = [f.path for f in os.scandir(
                path) if '.jpg' in f.name] if os.path.isdir(path) else [path]
            self.data = pd.DataFrame(data, columns=['image'])

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        image = Image.open(item.image).convert("RGB")

        if self.dataset_type is not PolycraftMappingDataset.DatasetType.PREDICT:
            grid = np.load(item.truth)

        if self.target_transforms:
            image, grid = self.target_transforms((image, grid))

        if self.transforms:
            image = self.transforms(image)

        if self.dataset_type == PolycraftMappingDataset.DatasetType.PREDICT:
            return item.image, image

        grid = torch.from_numpy(grid)

        return image, grid

    def __len__(self):
        return len(self.data)

    @staticmethod
    def args(parser):
        parser.add_argument("--path", type=str,
                            help="path to relevant data. this should be a csv file with pointers from image path to grid path (with headers image, truth) if training or validation, and an image or directory of image if predicting", required=True)
        parser.add_argument("--root_path", type=str, default='',
                            help="path to root of data if pointers are relative")

        super(PolycraftMappingDataset, PolycraftMappingDataset).args(parser)


class PolycraftMaskClassification(Dataset):

    def __init__(self,
                 path,
                 grayscale_transform,
                 horizontal_flip_transform,
                 color_jitter_transform,
                 rotate_transform,
                 **kwargs):

        super().__init__(**kwargs)

        transforms = []

        mask_transforms = [_transforms.ToTensor()]

        if self.dataset_type is Dataset.DatasetType.train:

            if grayscale_transform:
                transforms.insert(
                    0, _transforms.RandomGrayscale(p=grayscale_transform))
            if horizontal_flip_transform:
                mask_transforms.insert(
                    0, _transforms.RandomHorizontalFlip(p=horizontal_flip_transform))
            if color_jitter_transform:
                transforms.insert(0, _transforms.ColorJitter(
                    *color_jitter_transform))
            if rotate_transform:
                mask_transforms.append(_transforms.RandomRotation(
                    degrees=rotate_transform, expand=True))

        self.transforms = _transforms.Compose(transforms)
        self.mask_transforms = _transforms.Compose(mask_transforms)

        self.data = pd.read_csv(path)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        image = Image.open(item['image']).convert("RGB")

        image = self.transforms(image)

        mask = Image.fromarray(np.load(item['object']), mode='L')

        image.putalpha(mask)

        image = self.mask_transforms(image)

        return image, item['target'] - 1

    def __len__(self):
        return len(self.data)

    @staticmethod
    def args(parser):
        parser.add_argument("--path", type=str,
                            help="path to relevant data. this should be a csv file with pointers from image path to grid path (with headers image, truth) if training or validation, and an image or directory of image if predicting", required=True)

        parser.add_argument("--grayscale_transform", nargs='?', default=None, const=.1, type=float,
                            help="transform the input images to a grayscale at some rate (default: .1)")
        parser.add_argument("--horizontal_flip_transform", nargs='?', default=None, const=.5, type=float,
                            help="transform the input images and grids by horizontally flipping them at some rate (default: .5)")
        parser.add_argument("--rotate_transform", nargs='?',
                            default=None, const=2, type=int)
        parser.add_argument("--color_jitter_transform",
                            type=float, default=None, nargs='*')

        super(PolycraftMaskClassification,
              PolycraftMaskClassification).args(parser)
