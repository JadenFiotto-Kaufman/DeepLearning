from .torch import ImageFolder
import os

class ImageFolderSubset(ImageFolder):

    def __init__(self, exclude,  **kwargs):
        super().__init__(**kwargs)

        _include_class = 0
        _exclude_class = len(self.classes) - len(exclude)

        self.idx_to_subset_idx = {}

        for _class in self.class_to_idx:
            if _class in exclude:
                self.idx_to_subset_idx[self.class_to_idx[_class]] = _exclude_class
                _exclude_class += 1
            else:
                self.idx_to_subset_idx[self.class_to_idx[_class]] = _include_class
                _include_class += 1


        _include = []
        _exclude = []
        
        for sample in self.samples:
            sample_path, sample_class = sample
            sample_class = self.idx_to_subset_idx[sample_class]
            sample = (sample_path, sample_class)
            if sample_path.split(os.path.sep)[-2] in exclude:
                _exclude.append(sample)
            else:
                _include.append(sample)

        self.samples = _include
        self.excluded = _exclude

    @staticmethod
    def args(parser):
        parser.add_argument("--exclude", default=[], nargs='+')

        super(ImageFolderSubset,ImageFolderSubset).args(parser)

