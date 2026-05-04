import torch
import torch.nn as nn
import torchvision.models as models
from utils.rotation import rot6d_to_matrix

class customCNN(nn.Module):
    """5-layer CNN for single-channel depth processing."""
    def __init__(self, out_features=512):
        super(customCNN, self).__init__()
        self.features = nn.Sequential(
            # Layer 1: 224 -> 112
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            # Layer 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            # Layer 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            # Layer 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), 
            # Layer 5: 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(512, out_features)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))

class RGBD_FusionPredictor_custom(nn.Module):
    def __init__(self):
        super(RGBD_FusionPredictor_custom, self).__init__()
        # RGB branch (ResNet-50)
        self.rgb_backbone = models.resnet50(weights='IMAGENET1K_V1')
        self.rgb_backbone.fc = nn.Identity() 
        
        # Depth branch (custom 1-channel CNN)
        self.depth_backbone = customCNN(out_features=512)
        
        self.meta_encoder = nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        
        # Fusion MLP: 2048 (RGB) + 512 (Depth) + 64 (Meta) = 2624
        self.fusion_mlp = nn.Sequential(nn.Linear(2048 + 512 + 64, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.translation_head = nn.Linear(256, 3)
        self.rotation_head = nn.Linear(256, 6)     # 6D continuous representation (Zhou et al. 2019)

    def forward(self, rgb_crop, depth_crop, meta_info):
        f_rgb = self.rgb_backbone(rgb_crop)
        f_depth = self.depth_backbone(depth_crop)
        f_meta = self.meta_encoder(meta_info)
        shared = self.fusion_mlp(torch.cat((f_rgb, f_depth, f_meta), dim=1))
        return self.translation_head(shared), rot6d_to_matrix(self.rotation_head(shared))