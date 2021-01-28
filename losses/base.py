from torch.nn.modules.loss import _Loss
from DeepLearning.base import Base

class Loss(Base, _Loss):
    def __init__(self, model, **kwargs):
        self.model = model
        
        Base.__init__(self, **kwargs)
        _Loss.__init__(self)