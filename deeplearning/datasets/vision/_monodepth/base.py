from __future__ import absolute_import, division, print_function

import copy
import os
import random

import numpy as np
import PIL.Image as pil
import skimage.transform
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from deeplearning.datasets.base import Dataset

from .utils import generate_depth_map, pil_loader, readlines



class _MonoDataset(Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_ids
        num_scales
        is_train
        img_ext
    """

   

    def __init__(self,
                 data_path,
                 split,
                 image_size,
                 frame_ids,
                 use_stereo,
                 scales,
                 img_ext='.jpg',
                 **kwargs):

        super().__init__(**kwargs)

        frame_ids = frame_ids.copy()

        height, width = (image_size[0], image_size[0]) if len(image_size) == 1 else image_size

        assert height % 32 == 0, "'height' must be a multiple of 32"
        assert width % 32 == 0, "'width' must be a multiple of 32"

        self.height = height
        self.width = width

        self.data_path = data_path
        self.scales = scales
        self.use_stereo = use_stereo
        self.num_scales = len(scales)
        self.interp = Image.ANTIALIAS

        assert frame_ids[0] == 0, "frame_ids must start with 0"

        if use_stereo:
            frame_ids.append("s")

        self.frame_ids = frame_ids

        self.is_train = self.dataset_type is Dataset.DatasetType.train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        fpath = os.path.join(split, "{}_files.txt")
        filenames = readlines(fpath.format("train")) if self.is_train else readlines(fpath.format("val"))

        self.data = {}
        self.filenames = []
        for filename in filenames:
            config, frame, path, side = filename.split()
            frame = int(frame)

            if config not in self.data:
                self.data[config] = {}
            if frame not in self.data[config]:
                self.data[config][frame] = {}
            self.data[config][frame][side] = path



            if frame != 0:
                self.filenames.append(filename)
            elif self.filenames:
                del self.filenames[-1]
                del self.filenames[-1]
                
        del self.filenames[-1]
        del self.filenames[-1]
        del self.filenames[-1]
        del self.filenames[-1]


        random.shuffle(self.filenames)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        key = line[0]

        frame_index = int(line[1])
        side = line[3]


        for i in self.frame_ids:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(key, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(key, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_ids:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_ids:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs, torch.empty(1)

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    @staticmethod
    def args(parser):
        parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 required=True)
        parser.add_argument("--image_size", nargs='+', type=int, required=True)
        parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        parser.add_argument("--split",
                                 type=str,
                                 help="path to train/val split")

        super(_MonoDataset,_MonoDataset).args(parser)