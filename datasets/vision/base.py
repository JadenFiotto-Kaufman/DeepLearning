from torchvision.transforms import Compose
from torchvision.transforms import transforms as _transforms

from .. import Dataset


class _VisionDataset(Dataset):

    __normalization__ = None

    def __init__(
        self, 
        grayscale_transform,
        horizontal_flip_transform,
        color_jitter_transform,
        rotate_transform,
        normalization, 
        image_resize,
        image_center_crop,
        transforms = None,
        **kwargs):

        kwargs = super().__init__(**kwargs)

        transforms = transforms if transforms else []

        if self.dataset_type is Dataset.DatasetType.train:

            if grayscale_transform:
                transforms.insert(0, _transforms.RandomGrayscale(p=grayscale_transform))
            if horizontal_flip_transform:
                transforms.insert(0, _transforms.RandomHorizontalFlip(p=horizontal_flip_transform))
            if color_jitter_transform:
                transforms.insert(0, _transforms.ColorJitter(.4, .4, .4, .2))
            if rotate_transform:
                transforms.insert(0, _transforms.RandomRotation(degrees=rotate_transform, expand=True))

        if image_resize:
            image_resize = (image_resize[0], image_resize[0]) if len(image_resize) == 1 else image_resize
            transforms.append(_transforms.Resize(image_resize))
        if image_center_crop:
            image_center_crop = (image_center_crop[0], image_center_crop[0]) if len(image_center_crop) == 1 else image_center_crop
            transforms.append(_transforms.CenterCrop(image_center_crop))
        transforms.append(_transforms.ToTensor())
        if normalization:
            if len(normalization) == 6:
                transforms.append(_transforms.Normalize(normalization[:3], normalization[3:6]))
            if len(normalization) == 1 and self.__normalization__:
                transforms.append(_transforms.Normalize(self.__normalization__[0], self.__normalization__[1]))
        kwargs['transform'] = Compose(transforms)

        return kwargs

    @staticmethod
    def args(parser):
        parser.add_argument("--image_resize", type=int, nargs='+', required=True)
        parser.add_argument("--image_center_crop", type=int, default=None, nargs='*')

        parser.add_argument("--normalization", nargs='*', default=None,type=float)
        parser.add_argument("--grayscale_transform", nargs='?', default=None, const=.1, type=float,
                        help="transform the input images to a grayscale at some rate (default: .1)")
        parser.add_argument("--horizontal_flip_transform", nargs='?', default=None, const=.5, type=float,
                            help="transform the input images and grids by horizontally flipping them at some rate (default: .5)")
        parser.add_argument("--rotate_transform", nargs='?',
                        default=None, const=2, type=int)
        parser.add_argument("--color_jitter_transform",
                            default=False, const=True, nargs='?')

        super(_VisionDataset, _VisionDataset).args(parser)
