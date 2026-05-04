import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.rotation import rot6d_to_matrix


class ResidualBlock(nn.Module):
    """Basic residual block with skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResNet1ch(nn.Module):
    """Single-channel ResNet-like backbone for depth processing."""
    def __init__(self, out_features=512):
        super(ResNet1ch, self).__init__()
        
        self.start = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer2 = ResidualBlock(64, 128, stride=2)   # 112 -> 56
        self.layer3 = ResidualBlock(128, 256, stride=2)  # 56 -> 28
        self.layer4 = ResidualBlock(256, 512, stride=2)  # 28 -> 14
        self.layer5 = ResidualBlock(512, 512, stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_features)

    def forward(self, x):
        x = self.start(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        return self.fc(x.view(x.size(0), -1))


class RGBD_FusionPredictor_custom(nn.Module):
    """RGB-D fusion model with custom depth backbone."""
    def __init__(self):
        super(RGBD_FusionPredictor_custom, self).__init__()
        
        # RGB branch: pretrained ResNet-50
        self.rgb_backbone = models.resnet50(weights='IMAGENET1K_V1')
        num_features_rgb = self.rgb_backbone.fc.in_features  # 2048
        self.rgb_backbone.fc = nn.Identity() 
        
        # Depth branch: custom 1-channel ResNet
        self.depth_backbone = ResNet1ch(out_features=512)
        
        # Metadata encoder (bbox + camera info)
        self.meta_encoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Total: 2048 (RGB) + 512 (Depth) + 64 (Meta) = 2624
        combined_features = num_features_rgb + 512 + 64
        
        # Pose estimation MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.translation_head = nn.Linear(256, 3)
        self.rotation_head = nn.Linear(256, 6)     # 6D continuous representation (Zhou et al. 2019)

    def forward(self, rgb_crop, depth_crop, meta_info):
        f_rgb = self.rgb_backbone(rgb_crop)       # (B, 2048)
        f_depth = self.depth_backbone(depth_crop) # (B, 512)
        f_meta = self.meta_encoder(meta_info)     # (B, 64)

        fused = torch.cat((f_rgb, f_depth, f_meta), dim=1)
        shared = self.fusion_mlp(fused)

        translation = self.translation_head(shared)
        rotation = rot6d_to_matrix(self.rotation_head(shared))  # (B, 3, 3) in SO(3)

        return translation, rotation