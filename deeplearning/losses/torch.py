import torch
from torch.nn import CrossEntropyLoss as _CrossEntropyLoss

from .base import Loss


class _WeightedLoss(Loss):

    def __init__(self, **kwargs):
        Loss.__init__(self, **kwargs)
        kwargs['weight'] = torch.FloatTensor(kwargs['weight'])
        return kwargs

    @staticmethod
    def args(parser):
        parser.add_argument("--weight",
                                type=float,
                                 nargs='+',
                                 default=None)
        super(_WeightedLoss,_WeightedLoss).args(parser)

class CrossEntropyLoss(_WeightedLoss, _CrossEntropyLoss):

    def __init__(self, **kwargs):
        kwargs = _WeightedLoss.__init__(self, **kwargs)
        _CrossEntropyLoss.__init__(self, **kwargs)

