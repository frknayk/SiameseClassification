import torch
import torch.nn as nn
import torch.nn.functional as F
from models.models_utils import ImageClassificationBase, conv_block

class Siamese_v1(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 64, 10),  # 64@96*96
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 64@48*48
                nn.Conv2d(64, 128, 7),
                nn.ReLU(),    # 128@42*42
                nn.MaxPool2d(2),   # 128@21*21
                nn.Conv2d(128, 128, 4),
                nn.ReLU(), # 128@18*18
                nn.MaxPool2d(2), # 128@9*9
                nn.Conv2d(128, 256, 4),
                nn.ReLU(),   # 256@6*6
            )
        # num_params = count_parameters(self.conv)
        self.linear = nn.Sequential(nn.Linear(16384, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
            x = self.conv(x)
            x = x.view(x.size()[0], -1)
            x = self.linear(x)
            return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive