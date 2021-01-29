from .base import Model
from torch.nn import DataParallel as _DataParallel

class DataParallel(Model.__wrapper__, _DataParallel):

    def __init__(self, **kwargs):
        
        kwargs = Model.__wrapper__.__init__(self, **kwargs)
        _DataParallel.__init__(self, self._obj, **kwargs)

