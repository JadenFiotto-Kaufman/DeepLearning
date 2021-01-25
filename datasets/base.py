import argparse
from enum import Enum

import numpy as np
from DeepLearning.base import Base
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as _Dataset


class Dataset(Base, _Dataset):

    class DatasetType(Enum):
        train = 1,
        validate = 2,
        predict = 2

        def __str__(self):
            return self.name
        

    def __init__(self, dataset_type, batch_size, num_workers, cuda, **kwargs):
        Base.__init__(self, **kwargs)

        self.dataset_type = Dataset.DatasetType[dataset_type] if isinstance(dataset_type, str) else dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cuda = cuda

        return kwargs

    def dataloader(self):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle= self.dataset_type is Dataset.DatasetType.train,
            num_workers=self.num_workers,
            pin_memory=self.cuda
        )

    @staticmethod
    def args(parser):
        parser.add_argument("--dataset_type", type=str, choices=list(map(str, Dataset.DatasetType)), default=str(Dataset.DatasetType.train))
        parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size(default: 256)")
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--cuda", nargs='?', default=False, const=True, type=bool)

        super(Dataset,Dataset).args(parser)

    @staticmethod
    def val_args(cls):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        cls.args(parser)

        for action in parser._actions:
            print(action)
        
        return vars(parser.parse_known_args()[0])



class DatasetSplit(Dataset):

    def __init__(self, dataset, val_percent):
        
        self._dataset = dataset
        

        np.random.seed(42)
        indicies = np.arange(len(dataset))
        np.random.shuffle(indicies)

        self._indicies = indicies[int(len(dataset) * val_percent):] if dataset.dataset_type is Dataset.DatasetType.train else indicies[:int(len(dataset) * val_percent)]

    def dataloader(self):
        return DataLoader(
            self,
            batch_size=self._dataset.batch_size,
            shuffle= self._dataset.dataset_type is Dataset.DatasetType.train,
            num_workers=self._dataset.num_workers,
            pin_memory=self._dataset.cuda
        )

    def __len__(self):
        return len(self._indicies)


    def __getitem__(self, index):
        return self._dataset.__getitem__(self._indicies[index])



