from .torch import VGG16_bn as _VGG16_bn, VGG16 as _VGG16
import torch

class VGG16_bn(_VGG16_bn):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model.avgpool = torch.nn.AdaptiveAvgPool2d(1)

        self.model.classifier = torch.nn.Linear(512, self.model.classifier[-1].out_features, bias=True)


class VGG16(_VGG16):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model.avgpool = torch.nn.AdaptiveAvgPool2d(1)

        self.model.classifier = torch.nn.Linear(512, self.model.classifier[-1].out_features, bias=True)



