import argparse

import torch
from torchvision.transforms import transforms

from DeepLearning.datasets.vision.misc import ImageFolderSubset


def calculate_normalization(dataloader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print(mean)
    print(std)

if __name__ == "__main__":
    loader = ImageFolderSubset.get_instance(ImageFolderSubset).dataloader()

    calculate_normalization(loader)

   