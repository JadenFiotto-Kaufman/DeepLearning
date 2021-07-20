from deeplearning.validators.base import Validator
import torch

class BinaryClassificationAccuracy(Validator):


    def name(self):
        return "Accuracy"

    def format(self):
        return ':.2f'

    def __call__(self, output, targets):

        output = torch.sigmoid(output) > .5
        targets = targets > .5

        return sum(output == targets) / len(output)


class ClassificationAccuracy(Validator):

    def __init__(self, topn, **kwargs):

        super().__init__(**kwargs)

        self.topn = topn

    def name(self):
        return f"Accuracy top {self.topn}"

    def format(self):
        return ':.2f'

    def __call__(self, output, targets):

        output = torch.softmax(output, dim=1).argsort(dim=1)[:,-self.topn:]

        return sum([targets[i] in output[i] for i in range(len(output))]) / len(output)
        

    @staticmethod
    def args(parser):
        parser.add_argument('--topn', type=int, required=True)

        super(ClassificationAccuracy, ClassificationAccuracy).args(parser)