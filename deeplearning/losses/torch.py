from torch.nn import CrossEntropyLoss as _CrossEntropyLoss
from .base import Loss

class CrossEntropyLoss(Loss, _CrossEntropyLoss):

    def __init__(self, **kwargs):
        Loss.__init__(self, **kwargs)
        _CrossEntropyLoss.__init__(self, **kwargs)

