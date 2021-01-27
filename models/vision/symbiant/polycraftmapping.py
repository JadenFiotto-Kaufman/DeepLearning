import torch
from DeepLearning.models.base import Model

class Unflatten(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = tuple(shape)

    def __repr__(self):
        return f'Unflatten{self.shape}'

    def forward(self, x):
        x = x.view((x.shape[0], -1) + self.shape)
        return x


class MappingModel(Model):
    def __init__(self, backbone, n_classes, grid_size, param):

        assert grid_size in (133, 17)

        super(MappingModel, self).__init__()

        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        self.reduce = torch.nn.Conv2d(512,param, (1,1), stride=(1,1))

        self.fc = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(param*8*8, param*16),
                torch.nn.ReLU(),
                torch.nn.Linear(param*16, param * 8 * 8),
                torch.nn.ReLU(),
                Unflatten(8,8)
                )
        #TODO Make this piece more generic to grid_size if possible
        if grid_size == 133:
            self.increase = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(param, param // 2, (4,4), stride=(2,2), padding=(1,1)),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(param // 2,param // 4, (4,4), stride=(2,2), padding=(1,1), output_padding=1),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(param // 4, param // 8, (4,4), stride=(2,2), padding=(1,1)),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(param // 8, n_classes, (4,4), stride=(2,2), padding=(1,1), output_padding=1)
                    )
        elif grid_size == 17:
            self.increase = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(64, n_classes, (4,4), stride=(2,2), padding=(1,1), output_padding=1)
                    )

    def forward(self, x):

        x = self.backbone(x)
        x = self.reduce(x)
        x = self.fc(x)
        x = self.increase(x)
        
        return x 

