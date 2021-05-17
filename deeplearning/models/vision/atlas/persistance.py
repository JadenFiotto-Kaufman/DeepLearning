import torch
from deeplearning.models.vision.base import _VisionModel
from deeplearning.models.vision.pytorch.resnet import Resnet18
from deeplearning.models.vision.pytorch.vgg import VGG11_bn

class FAClassification(_VisionModel):

    def __init__(self, num_classes,columns, **kwargs):
        super().__init__(**kwargs)

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,kernel_size=(3,3)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(.2),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,kernel_size=(1,1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,kernel_size=(1,1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(.1)
        )

        n_feaures = {
            1 : 256,
            2 : 512,
            3 : 896,
            4 : 1152
        }


        self.fc = torch.nn.Sequential(
            torch.nn.Linear(n_feaures[len(columns)],64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64,num_classes)
        )

      
    @staticmethod
    def args(parser):
        parser.add_argument('--num_classes', type=int, required=True)
        parser.add_argument("--columns", nargs='+',type=str, required=True ,help="persistance image columns")

        super(FAClassification, FAClassification).args(parser)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x  


class FAClassificationResnet(Resnet18):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        _conv1 = self.model.conv1
        self.model.conv1 = torch.nn.Conv2d(1, _conv1.out_channels, kernel_size=_conv1.kernel_size, stride=_conv1.stride, padding=_conv1.padding,
                               bias=True)

        self.dropout20 = torch.nn.Dropout(.2)
        self.dropout50 = torch.nn.Dropout(.5)


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.dropout50(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)

        x = self.dropout50(x)

        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout50(x)

        x = self.model.fc(x)

        return x

        

class FAClassificationVGG(VGG11_bn):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model.features[0] = torch.nn.Conv2d(1, self.model.features[0].out_channels, kernel_size=self.model.features[0].kernel_size, stride=self.model.features[0].stride, padding=self.model.features[0].padding)

        self.dropout20 = torch.nn.Dropout(.2)
        self.dropout50 = torch.nn.Dropout(.5)

        self.model.classifier = torch.nn.Sequential(self.dropout50, self.model.classifier)



