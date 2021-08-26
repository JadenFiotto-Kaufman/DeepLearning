from torch.optim.lr_scheduler import StepLR as _StepLR, ReduceLROnPlateau as _ReduceLROnPlateau
from .base import Scheduler

class StepLR(Scheduler, _StepLR):

    def __init__(self, optimizer, **kwargs):
        _StepLR.__init__(self, optimizer, **kwargs)
        Scheduler.__init__(self, **kwargs)
        

    @staticmethod
    def args(parser):
        parser.add_argument("--step_size", type=int, default=5, required=True)
        parser.add_argument("--gamma", type=float, default=.1)

        super(StepLR,StepLR).args(parser)


class ReduceLROnPlateau(Scheduler, _ReduceLROnPlateau):

    def __init__(self, optimizer, **kwargs):
        _ReduceLROnPlateau.__init__(self, optimizer, **kwargs)
        Scheduler.__init__(self, **kwargs)
        

    @staticmethod
    def args(parser):
        parser.add_argument("--patience", type=int, default=5)
        parser.add_argument("--factor", type=float, default=.1)
        parser.add_argument("--threshold", type=float, default=.0001)

        super(ReduceLROnPlateau,ReduceLROnPlateau).args(parser)


