from .base import Model
from torch.nn.parallel import DataParallel as _DataParallel, DistributedDataParallel as _DistributedDataparallel

class DataParallel(Model.__wrapper__, _DataParallel):

    def __init__(self, **kwargs):
        
        kwargs = Model.__wrapper__.__init__(self, **kwargs)
        _DataParallel.__init__(self, self._obj, **kwargs)


    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)