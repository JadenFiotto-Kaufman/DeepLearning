from torch.nn import Module
from DeepLearning.base import Base

class Model(Base,Module):

    def __init__(self, **kwargs):
        Base.__init__(self, **kwargs)
        Module.__init__(self)

