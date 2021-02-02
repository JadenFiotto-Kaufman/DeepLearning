from DeepLearning import util
from DeepLearning import Base
from DeepLearning.models.vision.monodepth import DepthEstimator
from DeepLearning.datasets.torch import ImageFolder

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision.transforms import functional as F

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--load', type=str, required=True)

    parser.add_argument('--image', action='store_true')
    parser.add_argument("--cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):


    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        os.makedirs(os.path.join(args.out_path, 'image_folder_dataset'))

    for img in os.listdir(args.input_path):
        basename = os.path.splitext(img)[0]
        os.makedirs(os.path.join(args.out_path, 'image_folder_dataset', basename))

    device = util.get_device('cuda' if args.cuda else 'cpu')

    model_state_dict, checkpoint = util.load(args.load, device, dont_load_args=True)

    model = Base.get_instance(DepthEstimator)

    feed_width, feed_height = model.width, model.height

    dataset = Base.get_instance(ImageFolder)
    loader = dataset.dataloader()

   

    model.eval()
    with torch.no_grad():
        for i, (data, paths) in enumerate(loader):

            # Load image and preprocess
            original_width, original_height = data.shape[1:]
            input_image = input_image.resize((feed_width, feed_height), interpolation=pil.LANCZOS)

            # PREDICTION
            input_image = input_image.to(device)
            features = model.encoder(input_image)
            outputs = model.depth(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(  
                disp, (original_height, original_width), mode="bilinear", align_corners=False)


            for i, _data in enumerate(data):
                output_name = os.path.basename(os.path.normpath(paths[i]))
                name_dest_npy = "{}_disp.npy".format(output_name)
                scaled_disp, _ = disp_to_depth(_data, model.min_depth, model.max_depth)
                scaled_disp = scaled_disp.cpu().numpy()
                if not args.image:
                    np.save(os.path.join(args.out_path, 'depth', name_dest_npy), scaled_disp)
                else:
                    disp_resized_np = disp_resized.squeeze().cpu().numpy()
                    vmax = np.percentile(disp_resized_np, 95)
                    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                    im = pil.fromarray(colormapped_im)

                    name_dest_im =  "{}_disp.jpeg".format(output_name)
                    im.save(os.path.join(args.out_path, 'heat', name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
