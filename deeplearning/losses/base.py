from torch.nn.modules.loss import _Loss
from deeplearning.base import Base

class Loss(Base, _Loss):
    def __init__(self,**kwargs):  
        Base.__init__(self, **kwargs)
        _Loss.__init__(self)
