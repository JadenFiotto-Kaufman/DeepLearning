from .torch import ImageFolder
import os

class ImageFolderSubset(ImageFolder):

    def __init__(self, exclude,  **kwargs):
        super().__init__(**kwargs)

        _include = []
        _exclude = []
        
        for sample in self.samples:
            sample_path, sample_class = sample
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

