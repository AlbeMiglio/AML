import torch
import torch.nn as nn
import torchvision.models as models
from utils.rotation import rot6d_to_matrix

class RGBD_FusionPredictor(nn.Module):
    def __init__(self):
        super(RGBD_FusionPredictor, self).__init__()
        
        # RGB branch: ResNet-50
        self.rgb_backbone = models.resnet50(weights='IMAGENET1K_V1')
        num_features_rgb = self.rgb_backbone.fc.in_features  # 2048
        self.rgb_backbone.fc = nn.Identity() 
        
        # Depth branch: ResNet-18
        self.depth_backbone = models.resnet18(weights='IMAGENET1K_V1')
        num_features_depth = self.depth_backbone.fc.in_features  # 512
        self.depth_backbone.fc = nn.Identity()
        
        # Metadata branch (camera info and bbox): [cx, cy, w, h, fx, fy, px, py]
        self.meta_encoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Total: 2048 (RGB) + 512 (Depth) + 64 (Meta) = 2624
        combined_features = num_features_rgb + num_features_depth + 64
        
        # Pose Estimator (MLP)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.translation_head = nn.Linear(256, 3)  # X, Y, Z
        self.rotation_head = nn.Linear(256, 6)     # 6D continuous representation (Zhou et al. 2019)

    def forward(self, rgb_crop, depth_crop, meta_info):
        """
        rgb_crop: (B, 3, 224, 224)
        depth_crop: (B, 3, 224, 224)
        meta_info: (B, 8) - normalized [cx, cy, w, h, fx, fy, px, py]
        """
        f_rgb = self.rgb_backbone(rgb_crop)       # 2048
        f_depth = self.depth_backbone(depth_crop) # 512
        f_meta = self.meta_encoder(meta_info)     # 64

        # Feature fusion by concatenation
        fused = torch.cat((f_rgb, f_depth, f_meta), dim=1)  # 2624
        shared = self.fusion_mlp(fused)

        translation = self.translation_head(shared)
        rotation = rot6d_to_matrix(self.rotation_head(shared))  # (B, 3, 3) in SO(3)

        return translation, rotation