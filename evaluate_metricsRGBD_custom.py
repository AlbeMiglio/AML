import torch
import numpy as np
import cv2
import os
import yaml
import trimesh
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Inference pipeline (matches val_transform from dataset)
inference_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'depth': 'mask'})



# Scegli quale modello usare commentando/scommentando la riga appropriata:
#from models.RGBD_FusionPredictor_custom import RGBD_FusionPredictor_custom  # 5-layer CNN
from models.FusionResNetCustom import RGBD_FusionPredictor_custom  # ResNet10 custom




# Selezione automatica del path dei pesi in base al modello importato
if "FusionResNetCustom" in RGBD_FusionPredictor_custom.__module__:
    MODEL_PATH = "weights/resnet10_custom/pose_rgbd_custom_1ch_best.pth"
    print("Using ResNet-10 weights.")
else:
    MODEL_PATH = "weights/5layer_cnn/pose_rgbd_custom_1ch_best.pth"
    print("Using 5-layer CNN weights.")

from data.split import prepare_data_and_splits
from utils.rgbd_utils_custom import (
    load_info_cache,
    fetch_sample_info,
    convert_depth_to_meters,
    square_crop_coords,
    build_meta_tensor,
)
from utils.rgbd_utils import get_object_metadata, select_detection_for_object
from utils.pose_metrics import pose_error


def get_annotation(gt_cache, obj_id, sample_id):
    ann_list = gt_cache[obj_id][sample_id]
    return next((ann for ann in ann_list if ann['obj_id'] == obj_id), ann_list[0])


def generate_fusion_report(test_samples, gt_cache, info_cache, model, yolo_model, models_info, root_dataset, device):
    class_names = {
        1: "ape", 2: "benchvise", 3: "bowl", 4: "camera", 5: "can",
        6: "cat", 7: "cup", 8: "driller", 9: "duck", 10: "eggbox",
        11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"
    }

    ply_cache = {}
    for oid in class_names:
        ply_path = os.path.join(root_dataset, 'models', f'obj_{oid:02d}.ply')
        if os.path.exists(ply_path):
            ply_cache[oid] = trimesh.load(ply_path).sample(500)

    results = []
    model.eval()
    detection_misses = 0
    missing_info = 0
    processed = 0

    print("\nEvaluating RGB-D Custom model...")
    with torch.no_grad():
        for obj_id, sample_id in tqdm(test_samples):
            if obj_id not in class_names or obj_id not in ply_cache:
                continue

            ann = get_annotation(gt_cache, obj_id, sample_id)
            info_entry = fetch_sample_info(info_cache, obj_id, sample_id)
            if info_entry is None:
                missing_info += 1
                continue

            model_info = get_object_metadata(models_info, obj_id)
            if model_info is None:
                missing_info += 1
                continue

            img_path = os.path.join(root_dataset, 'data', f"{obj_id:02d}", 'rgb', f"{sample_id:04d}.png")
            depth_path = os.path.join(root_dataset, 'data', f"{obj_id:02d}", 'depth', f"{sample_id:04d}.png")

            img_bgr = cv2.imread(img_path)
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None or depth_raw is None:
                missing_info += 1
                continue

            depth_meters = convert_depth_to_meters(depth_raw, info_entry.get('depth_scale', 1.0))
            cam_K = np.array(info_entry['cam_K'], dtype=np.float32).reshape(3, 3)

            result = yolo_model(img_bgr, verbose=False)[0]
            box = select_detection_for_object(result, obj_id)
            if box is None:
                detection_misses += 1
                continue

            xyxy = box.xyxy.cpu().numpy()[0]
            yolo_bbox = [
                float(xyxy[0]), float(xyxy[1]),
                float(xyxy[2] - xyxy[0]), float(xyxy[3] - xyxy[1])
            ]

            crop_coords = square_crop_coords(yolo_bbox, img_bgr.shape)
            if crop_coords is None:
                detection_misses += 1
                continue

            left, top, right, bottom = crop_coords
            rgb_crop = cv2.resize(img_bgr[top:bottom, left:right], (224, 224))
            rgb_crop = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB)
            depth_crop = cv2.resize(depth_meters[top:bottom, left:right], (224, 224), interpolation=cv2.INTER_NEAREST)

            transformed = inference_transform(image=rgb_crop, depth=depth_crop)

            rgb = transformed['image'].unsqueeze(0).to(device)
            depth = transformed['depth'].float().unsqueeze(0).unsqueeze(0).to(device)
            meta = build_meta_tensor(yolo_bbox, cam_K, img_bgr.shape).to(device)

            pred_T, pred_R = model(rgb, depth, meta)
            R_pred = pred_R[0].cpu().numpy()         # (3, 3) from SO(3)
            T_pred = pred_T[0].cpu().numpy() * 1000.0  # metres -> mm

            R_gt = np.array(ann['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
            T_gt = np.array(ann['cam_t_m2c'], dtype=np.float32)
            points = ply_cache[obj_id]
            diameter = model_info['diameter']
            threshold = 0.10 * diameter

            add_score = pose_error(points, R_gt, T_gt, R_pred, T_pred, obj_id)
            results.append({
                "Classe": class_names[obj_id],
                "ADD_mm": add_score,
                "Success": add_score < threshold
            })
            processed += 1

    df = pd.DataFrame(results)
    if df.empty:
        print("No samples evaluated. Check YOLO detections or missing data.")
        print(f"Samples skipped (missing info): {missing_info}")
        print(f"Samples skipped (YOLO/crop): {detection_misses}")
        return

    summary = df.groupby("Classe").agg(
        Media_ADD=("ADD_mm", "mean"),
        Accuracy=("Success", lambda x: x.mean() * 100)
    ).reset_index()

    print("\n" + "=" * 60)
    print(f"{'CLASSE':<15} | {'ADD/ADD-S (mm)':<15} | {'ACCURACY (%)':<12}")
    print("-" * 60)
    for _, row in summary.iterrows():
        sym = "*" if row['Classe'] in ("eggbox", "glue") else " "
        print(f"{row['Classe']:<14}{sym}| {row['Media_ADD']:<15.2f} | {row['Accuracy']:<12.1f}")
    print("=" * 60)
    print("* ADD-S metric (symmetric objects)")
    print(f"GLOBAL AVERAGE -> Accuracy (ADD/ADD-S < 0.1d): {summary['Accuracy'].mean():.1f}%")
    total_samples = len([item for item in test_samples if item[0] in class_names])
    print(f"Samples processed: {processed}/{total_samples}")
    print(f"Samples skipped (missing info): {missing_info}")
    print(f"Samples skipped (YOLO/crop): {detection_misses}")


def main():
    ROOT_DATASET = "datasets/linemod/Linemod_preprocessed"
    YOLO_PATH = "weights/yolo/best.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using weights: {MODEL_PATH}")

    with open(os.path.join(ROOT_DATASET, 'models', 'models_info.yml'), 'r') as f:
        models_info = yaml.safe_load(f)

    _, _, test_samples, gt_cache = prepare_data_and_splits(ROOT_DATASET)
    info_cache = load_info_cache(ROOT_DATASET, sorted(gt_cache.keys()))

    model = RGBD_FusionPredictor_custom().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    yolo_model = YOLO(YOLO_PATH)

    generate_fusion_report(test_samples, gt_cache, info_cache, model, yolo_model, models_info, ROOT_DATASET, DEVICE)


if __name__ == "__main__":
    main()
