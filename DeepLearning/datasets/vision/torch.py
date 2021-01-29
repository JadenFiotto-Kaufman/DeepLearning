from torchvision.datasets import ImageFolder as _ImageFolder

from .base import _VisionDataset


class ImageFolder(_VisionDataset, _ImageFolder):

    def __init__(self, **kwargs):
        kwargs = _VisionDataset.__init__(self, **kwargs)
        _ImageFolder.__init__(self, **kwargs)

    @staticmethod
    def args(parser):    
        parser.add_argument("--root", type=str, required=True)

        super(ImageFolder,ImageFolder).args(parser)
