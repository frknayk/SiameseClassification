import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from .models_utils import ImageClassificationBase


class InceptionV3(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.model = models.inception_v3(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Parameters of newly constructed modules have requires_grad=True by default
        # Handle the auxilary net
        num_ftrs = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_ftrs, 4)
        # Handle the primary net
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 4)
        # print(self.model)

    def forward(self, xb):
        out = self.model(xb)
        return out



