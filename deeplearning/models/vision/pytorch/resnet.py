from deeplearning.models.vision.base import _VisionModel
from torchvision.models import resnet18, resnet50

class _Resnet(_VisionModel):

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)

        self.model = model

    @staticmethod
    def args(parser):
        parser.add_argument('--num_classes', type=int, required=True)

        super(_Resnet, _Resnet).args(parser)

    def forward(self, x):
        return self.model(x)  
    

class Resnet18(_Resnet):

    def __init__(self, **kwargs):
        model = resnet18(**kwargs)
        super().__init__(model=model,**kwargs)


class Resnet50(_Resnet):

    def __init__(self, **kwargs):
        model = resnet50(**kwargs)
        super().__init__(model=model,**kwargs)