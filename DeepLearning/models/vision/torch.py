from .base import _VisionModel
from torchvision.models import vgg16_bn, vgg16
import torch


class _Wrapper(_VisionModel):

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)

        self.model = model
    
    def forward(self, x):
        return self.model(x)

    


class _VGG(_Wrapper):

    @staticmethod
    def args(parser):
        parser.add_argument('--pretrained', nargs='?', default=False, const=True, type=bool)
        parser.add_argument('--num_classes', type=int, required=True)

        super(_VGG, _VGG).args(parser)
        

class VGG16_bn(_VGG):

    def __init__(self, **kwargs):
        model = vgg16_bn(**kwargs)
        super().__init__(model=model,**kwargs)

class VGG16(_VGG):

    def __init__(self, **kwargs):
        model = vgg16(**kwargs)
        super().__init__(model=model,**kwargs)