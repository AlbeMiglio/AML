import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PosePredictor(nn.Module):
    """ResNet-50 based pose predictor: unit quaternion + translation vector."""
    def __init__(self):
        super(PosePredictor, self).__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.regression_head = nn.Linear(num_features, 4)  # unit quaternion (x, y, z, w)
        self.tvec_head       = nn.Linear(num_features, 3)  # translation in meters

    def forward(self, x):
        features = self.backbone(x)
        quat = F.normalize(self.regression_head(features), p=2, dim=1)
        tvec = self.tvec_head(features)
        return quat, tvec