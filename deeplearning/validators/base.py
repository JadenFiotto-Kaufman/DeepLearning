from deeplearning.base import Base

class Validator(Base):

    def __init__(self, dataset, **kwargs):

        super().__init__(**kwargs)

        self.dataset = dataset

    def name(self):
        pass
    def format(self):
        pass
    def __call__(self, output, targets):
        pass