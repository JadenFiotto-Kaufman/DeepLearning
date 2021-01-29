from torch.nn import Module
from DeepLearning.base import Base

class Model(Base,Module):

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        
        Base.__init__(self, **kwargs)
        Module.__init__(self)

        return kwargs


class _ModelWrapper(Base.__wrapper__, Model):
    
    def __init__(self, **kwargs):
        kwargs = Model.__init__(self, **kwargs)
        kwargs = Base.__wrapper__.__init__(self, **kwargs)

        return kwargs


Model.__wrapper__ = _ModelWrapper