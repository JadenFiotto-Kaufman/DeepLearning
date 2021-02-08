from .base import Dataset
import numpy as np

class TrainValSplit(Dataset.__wrapper__):

    def __init__(self, val_percent, **kwargs):
        
        super().__init__(**kwargs)
        
        np.random.seed(42)
        indicies = np.arange(len(self._obj))
        np.random.shuffle(indicies)

        self._indicies = indicies[int(len(self._obj) * val_percent):] if self._obj.dataset_type is Dataset.DatasetType.train else indicies[:int(len(self._obj) * val_percent)]

    def __len__(self):
        return len(self._indicies)

    @staticmethod
    def args(parser):
        parser.add_argument("--val_percent",  default=.15, type=float)
        
        super(TrainValSplit,TrainValSplit).args(parser)




