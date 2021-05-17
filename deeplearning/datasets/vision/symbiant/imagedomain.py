import os
from random import random

import numpy as np
import pandas as pd
import torch
from deeplearning.datasets.base import Dataset
from PIL import Image
from torchvision.transforms import transforms 
from glob import glob
import albumentations


class TypeThreeNovelty(Dataset):

    novelties = {
        'blur' : albumentations.augmentations.transforms.Blur(always_apply=True),
        'rain' : albumentations.augmentations.transforms.RandomRain(always_apply=True),
        'snow' : albumentations.augmentations.transforms.RandomSnow(always_apply=True),
        'fog' : albumentations.augmentations.transforms.RandomFog(always_apply=True)
        }

    def __init__(self,
                 paths,
                 novelties,
                 image_resize,
                 **kwargs):

        super().__init__(**kwargs)

        if len(paths) > 1:
            path = paths[0] if self.dataset_type == Dataset.DatasetType.train else paths[1]
        else:
            path = paths[0]


        self.data = glob(os.path.join(path, '**/*.JPEG'))
        self.transforms = albumentations.OneOf([TypeThreeNovelty.novelties[novelty] for novelty in novelties])
        post_transforms = []
        image_resize = (image_resize[0], image_resize[0]) if len(image_resize) == 1 else image_resize
        if image_resize:
            post_transforms.append(transforms.Resize(image_resize))
        post_transforms.append(transforms.ToTensor())
        post_transforms.append(transforms.Normalize((.5, .5, .5), (.225, .225, .225)))
        self.post_transforms = transforms.Compose(post_transforms)



    def __getitem__(self, idx):
        image_path = self.data[idx]

        image = Image.open(image_path).convert("RGB")

        target = 1 if random() > .5 else 0

        if target:
            image = Image.fromarray(self.transforms(image=np.array(image))['image'])

        image = self.post_transforms(image)

        return image, float(target)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def args(parser):
        parser.add_argument("--paths", type=str, nargs='+', help="paths to relevant data. if 2 are given the first is train the second is validation", required=True)

        parser.add_argument("--novelties", nargs='+',type=str, choices=TypeThreeNovelty.novelties.keys(),help="type three novelties to train on")
        parser.add_argument("--image_resize", type=int, nargs='*', default=None)

        super(TypeThreeNovelty,
              TypeThreeNovelty).args(parser)
