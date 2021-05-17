import torch
from deeplearning.models.base import Model
from deeplearning.models.vision.pytorch.resnet import Resnet50
from deeplearning import util

class _Resnet50(Resnet50):

    def forward(self, x):
        conv1 = self.model.conv1(x)
        x = self.model.bn1(conv1)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        layer1 = self.model.layer1(x)
        # x = self.model.layer2(x)
        # x = self.model.layer3(x)
        # x = self.model.layer4(x)

        # x = self.model.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.model.fc(x)

        return conv1, layer1

class TypeThreeNoveltyClassifier(Model):

    def __init__(self, classifier_path, **kwargs):
        super().__init__(**kwargs)

        self.classifier = _Resnet50(num_classes=413)
        self.classifier.model.load_state_dict(torch.load(classifier_path))
        self.classifier.requires_grad_(False)

        in_features = 8512

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(in_features // 2),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, 1),

        )

    def stats(self, x):
        channel_means = torch.mean(x, axis=(2,3))
        channel_covars = self.cov(x.reshape((x.shape[0], 64, -1)))
        return channel_means, channel_covars

    def cov(self, m, rowvar=False):
        # if m.dim() > 2:
        #     raise ValueError('m has more than 2 dimensions')
        # if m.dim() < 2:
        #     m = m.view(1, -1)
        # if not rowvar and m.size(0) != 1:
        #     m = m.t()
        fact = 1.0 / (m.size(2) - 1)
        m -= torch.mean(m, dim=2, keepdim=True)
        mt = torch.transpose(m,1,2)
        return fact * m.matmul(mt)

    def forward(self, x):
        layers = self.classifier(x)

        results = []

        for layer in layers:

            means, covars = self.stats(layer)
            
            results.append(torch.cat((means, torch.reshape(covars, (covars.shape[0], covars.shape[1] ** 2))), dim=1))      
        
        results = torch.cat(results, dim=1)

        return self.fc(results).squeeze()

       
    @staticmethod
    def args(parser):
        parser.add_argument("--classifier_path", type=str,
                            help="path to pretrained resnet 50 classifier", required=True)

        super(TypeThreeNoveltyClassifier,
            TypeThreeNoveltyClassifier).args(parser)