from enum import Enum

from deeplearning.base import Base
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as _Dataset
from deeplearning.util import str2bool



class Dataset(Base, _Dataset):

    class DatasetType(Enum):
        train = 1,
        validate = 2,
        predict = 2

        def __str__(self):
            return self.name
        

    def __init__(self, dataset_type, batch_size, num_workers, drop_last, device, **kwargs):
        Base.__init__(self, **kwargs)

        self.dataset_type = Dataset.DatasetType[dataset_type] if isinstance(dataset_type, str) else dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.drop_last = drop_last

        return kwargs

    def dataloader(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle= self.dataset_type is Dataset.DatasetType.train,
            num_workers=self.num_workers,
            pin_memory='cuda' in self.device.type,
            drop_last=self.drop_last
        )

    @staticmethod
    def args(parser):
        parser.add_argument("--dataset_type", type=str, choices=list(map(str, Dataset.DatasetType)), default=str(Dataset.DatasetType.train))
        parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size(default: 256)")
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--drop_last", type=str2bool, nargs='?',
                        const=True, default=False)

        super(Dataset,Dataset).args(parser)


class _DatasetWrapper(Base.__wrapper__, Dataset):
    pass


Dataset.__wrapper__ = _DatasetWrapper