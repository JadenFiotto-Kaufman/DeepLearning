from torch.optim.lr_scheduler import StepLR as _StepLR
from .base import Scheduler

class StepLR(Scheduler, _StepLR):

    def __init__(self, optimizer, **kwargs):
        Scheduler.__init__(self, **kwargs)
        _StepLR.__init__(self, optimizer, **kwargs)

    @staticmethod
    def args(parser):
        parser.add_argument("--step_size", type=int, default=5, required=True)
        parser.add_argument("--gamma", type=float, default=.1)

        super(StepLR,StepLR).args(parser)


