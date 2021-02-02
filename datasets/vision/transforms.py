import torch
from torchvision.transforms import functional as F

class Crop(torch.nn.Module):

    def __init__(self, top, left, height, width):
        super().__init__()

        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, img):

        return F.crop(img, self.top, self.left, self.height, self.width)