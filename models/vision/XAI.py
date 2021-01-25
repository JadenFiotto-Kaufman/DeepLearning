from .torch import VGG16_bn as _VGG16_bn
import torch

class VGG16_bn(_VGG16_bn):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        in_features = self.model.classifier[0].in_features
        out_features = self.model.classifier[-1].out_features

        self.model.classifier = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=in_features, out_features=out_features)
        )

