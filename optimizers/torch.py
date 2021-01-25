from torch.optim import Adam as _Adam
from .base import Optimizer

class Adam(Optimizer, _Adam):

    def __init__(self, params, **kwargs):
        Optimizer.__init__(self, **kwargs)
        _Adam.__init__(self, params, **kwargs)

    @staticmethod
    def args(parser):
        parser.add_argument("--lr", type=float, default=.003,
                        help="learning rate for optimizer")
        parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="weight decay for optimizer ")

        super(Adam,Adam).args(parser)