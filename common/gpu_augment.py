import torch
import torch.nn as nn
import kornia.augmentation as K


class GPUAugmentation(nn.Module):
    """Photometric augmentation + ImageNet normalization applied on GPU.

    Replaces the per-sample albumentations pipeline (ColorJitter + GaussNoise
    + Normalize) that used to run inside Dataset.__getitem__. Moving it to GPU
    eliminates the CPU dataloader bottleneck.
    """

    def __init__(self):
        super().__init__()
        self.augment = K.AugmentationSequential(
            K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
            K.RandomGaussianNoise(mean=0.0, std=0.02, p=0.2),
            data_keys=["image"],
            same_on_batch=False,
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, rgb_uint8: torch.Tensor, training: bool) -> torch.Tensor:
        rgb = rgb_uint8.float() / 255.0
        if training:
            rgb = self.augment(rgb)
        return (rgb - self.mean) / self.std
