import torch
from torch.utils.data import Dataset
import os
import yaml
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

from phase3_baseline.losses import matrix_to_quaternion

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _build_transform(augment: bool):
    base = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if not augment:
        return transforms.Compose(base)

    aug = [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
    ]
    # RandomErasing works on tensors, so it goes after ToTensor/Normalize
    return transforms.Compose(aug + base + [transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3))])


class LineModDataset(Dataset):
    def __init__(self, dataset_root, samples, gt_cache, img_size=(224, 224), augment=False):
        self.dataset_root = dataset_root
        self.samples = samples      # Lista di (obj_id, img_id)
        self.gt_cache = gt_cache    # Cache dei GT
        self.img_size = img_size
        
        self.transform = _build_transform(augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
            obj_id, img_id = self.samples[idx]
            
            obj_folder = f"{obj_id:02d}"
            img_name = f"{img_id:04d}.png"
            img_path = os.path.join(self.dataset_root, 'data', obj_folder, 'rgb', img_name)
            
            img = Image.open(img_path).convert("RGB")
            
            ann_list = self.gt_cache[obj_id][img_id]
            target_ann = next((ann for ann in ann_list if ann['obj_id'] == obj_id), ann_list[0])
                
            x, y, w, h = target_ann['obj_bb'] 
            
            # Square crop to maintain aspect ratio for ResNet
            center_x = x + w / 2
            center_y = y + h / 2
            side = max(w, h)
            
            left = center_x - side / 2
            top = center_y - side / 2
            right = center_x + side / 2
            bottom = center_y + side / 2
            
            img_crop = img.crop((left, top, right, bottom))
            img_resized = img_crop.resize(self.img_size, Image.BILINEAR)
            img_tensor = self.transform(img_resized)
            
            R_mat = torch.tensor(target_ann['cam_R_m2c'], dtype=torch.float32).view(3, 3)
            T = torch.tensor(target_ann['cam_t_m2c'], dtype=torch.float32)
            
            quaternion_gt = matrix_to_quaternion(R_mat)
            
            return {
                "rgb": img_tensor,
                "quaternion": quaternion_gt,
                "R": R_mat,
                "T": T,
                "bbox_originale": torch.tensor([x, y, w, h]),
                "obj_id": obj_id,
                "sample_id": img_id
            }