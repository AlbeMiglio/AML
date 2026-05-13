import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import trimesh

from phase4_fusion.main.rgbd_utils import (
    convert_depth_to_meters,
    square_crop_coords,
    build_meta_tensor,
)

class LineModDatasetRGBD(Dataset):
    def __init__(self, dataset_root, samples, gt_cache, info_cache, img_size=(224, 224), n_points=500, is_train=False):
        self.dataset_root = dataset_root
        self.samples = samples
        self.gt_cache = gt_cache
        self.info_cache = info_cache
        self.img_size = img_size
        self.n_points = n_points
        self.is_train = is_train

        self.model_points_cache = {}
        unique_obj_ids = set([s[0] for s in samples])
        for obj_id in unique_obj_ids:
            self.model_points_cache[obj_id] = self._pre_load_model_points(obj_id)


    def _pre_load_model_points(self, obj_id):
            """Load .ply model and sample points in meters."""
            obj_folder = f"{obj_id:02d}"
            ply_path = os.path.join(self.dataset_root, 'models',  f"obj_{obj_folder}.ply")
            
            mesh = trimesh.load(ply_path)
            points = mesh.vertices
            
            if len(points) > self.n_points:
                idx = np.random.choice(len(points), self.n_points, replace=False)
                points = points[idx]
            
            # Convert mm to meters for consistency with depth
            return torch.from_numpy(points).float() / 1000.0
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_id, img_id = self.samples[idx]
        obj_folder = f"{obj_id:02d}"
        img_name = f"{img_id:04d}.png"
        
        ann_list = self.gt_cache[obj_id][img_id]
        target_ann = next((item for item in ann_list if item['obj_id'] == obj_id), ann_list[0])
        target_info = self.info_cache[obj_id][img_id]
        
        depth_scale = target_info['depth_scale']
        K = np.array(target_info['cam_K'], dtype=np.float32).reshape(3, 3)

        x, y, w, h = target_ann['obj_bb']
        
        rgb_path = os.path.join(self.dataset_root, 'data', obj_folder, 'rgb', img_name)
        depth_path = os.path.join(self.dataset_root, 'data', obj_folder, 'depth', img_name)
        
        rgb_img = cv2.imread(rgb_path)
        depth_img_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if rgb_img is None or depth_img_raw is None:
            raise FileNotFoundError(f"Missing images for obj {obj_id} img {img_id}")

        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        depth_meters = convert_depth_to_meters(depth_img_raw, depth_scale)

        crop_coords = square_crop_coords([x, y, w, h], rgb_img.shape)
        l, t, r, b = crop_coords
        
        rgb_crop = rgb_img[t:b, l:r]
        depth_crop = depth_meters[t:b, l:r]

        rgb_crop = cv2.resize(rgb_crop, self.img_size, interpolation=cv2.INTER_LINEAR)
        depth_crop = cv2.resize(depth_crop, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Raw uint8 RGB (3,H,W) — normalize + augment happen on GPU in train loop.
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb_crop.transpose(2, 0, 1)))  # uint8
        # Single-channel float depth (1,H,W) — replicated to 3ch on GPU before model.
        depth_tensor = torch.from_numpy(depth_crop).float().unsqueeze(0)

        meta_tensor = build_meta_tensor([x, y, w, h], K, rgb_img.shape)
        meta_info = meta_tensor.squeeze(0)
        
        # 7. Target della Posa e Modello 3D
        R_mat = torch.tensor(target_ann['cam_R_m2c'], dtype=torch.float32).view(3, 3)
        T_vec = torch.tensor(target_ann['cam_t_m2c'], dtype=torch.float32) / 1000.0
        model_points = self.model_points_cache[obj_id]
        
        return {
            "rgb": rgb_tensor,
            "depth": depth_tensor,
            "meta_info": meta_info,
            "rotation_9d": R_mat.flatten(),
            "translation_3d": T_vec,
            "obj_id": obj_id,
            "R_matrix": R_mat,
            "model_points": model_points
        }