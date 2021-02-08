
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import PIL.Image as pil
import skimage.transform

from ._monodepth.utils import generate_depth_map, pil_loader
from ._monodepth.base import _MonoDataset


class _HUGADataset(_MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, focal_length, image_full_size, **kwargs):
        super().__init__(**kwargs)

        image_full_size_height, image_full_size_width = (image_full_size[0], image_full_size[0]) if len(image_full_size) == 1 else image_full_size

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.K = np.array([[focal_length / self.width, 0, (self.width // 2.0 + (self.width - image_full_size_width)) / self.width, 0],
                           [0, focal_length / self.height, (self.height // 2.0 + (self.height - image_full_size_height)) / self.height, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.focal_length = focal_length
        self.full_res_shape = image_full_size
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        #color = color.crop((0, 0, 256, 224))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    @staticmethod
    def args(parser):
        parser.add_argument("--focal_length", type=float, required=True)
        parser.add_argument("--image_full_size", nargs='+', type=int, required=True)

        super(_HUGADataset,_HUGADataset).args(parser)


class HUGA(_HUGADataset):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def get_image_path(self, folder, frame_index, side):
        try:
            return self.data[folder][frame_index][side]
        except KeyError:
            print(self.data[folder])
            print(f'{folder} {frame_index} {side}')

            raise KeyError()
            

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
