import os
import csv
import argparse
from ultralytics import YOLO

DATA_YAML  = 'datasets/linemod/linemod_yolo_format/data.yaml'
MODEL_PATH = 'weights/yolo/best.pt'
RESULTS_DIR = 'results'

CLASS_NAMES = {
    0: "Ape", 1: "Benchvise", 2: "Bowl", 3: "Cam", 4: "Can",
    5: "Cat", 6: "Cup", 7: "Driller", 8: "Duck", 9: "Eggbox",
    10: "Glue", 11: "Holepuncher", 12: "Iron", 13: "Lamp", 14: "Phone"
}

def parse_opt():
    parser = argparse.ArgumentParser(description='Evaluate YOLO on LineMod test split')
    parser.add_argument('--model',  type=str, default=MODEL_PATH, help='Path to YOLO weights')
    parser.add_argument('--data',   type=str, default=DATA_YAML,  help='Path to data.yaml')
    parser.add_argument('--imgsz',  type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--split',  type=str, default='test',     help='Dataset split to evaluate (test/val)')
    return parser.parse_args()

def main(opt):
    if not os.path.exists(opt.model):
        print(f"Model not found: {opt.model}")
        return
    if not os.path.exists(opt.data):
        print(f"data.yaml not found: {opt.data}")
        print("Run: python data/prepare_yolo_data.py")
        return

    model = YOLO(opt.model)
    print(f"Evaluating on split='{opt.split}'...")

    metrics = model.val(
        data=opt.data,
        split=opt.split,
        imgsz=opt.imgsz,
        device=opt.device,
        verbose=False,
    )

    # Global metrics
    map50    = metrics.box.map50
    map5095  = metrics.box.map
    mp       = metrics.box.mp
    mr       = metrics.box.mr

    print("\n" + "=" * 65)
    print(f"{'YOLO Evaluation — split: ' + opt.split:<65}")
    print("=" * 65)
    print(f"{'Global mAP@0.5':<30} {map50 * 100:.2f}%")
    print(f"{'Global mAP@0.5:0.95':<30} {map5095 * 100:.2f}%")
    print(f"{'Mean Precision':<30} {mp * 100:.2f}%")
    print(f"{'Mean Recall':<30} {mr * 100:.2f}%")

    # Per-class metrics
    ap50_per_class   = metrics.box.ap50        # shape (num_classes,)
    ap5095_per_class = metrics.box.ap          # shape (num_classes,)
    class_indices    = metrics.box.ap_class_index  # which class indices were detected

    print("\n" + "-" * 65)
    print(f"{'Class':<15} {'AP@0.5':>10} {'AP@0.5:0.95':>14}")
    print("-" * 65)

    rows = []
    for i, cls_idx in enumerate(class_indices):
        cls_idx = int(cls_idx)
        name   = CLASS_NAMES.get(cls_idx, f"class_{cls_idx}")
        ap50   = float(ap50_per_class[i])
        ap5095 = float(ap5095_per_class[i])
        print(f"{name:<15} {ap50 * 100:>9.2f}% {ap5095 * 100:>13.2f}%")
        rows.append({"class": name, "AP50": round(ap50, 4), "AP50_95": round(ap5095, 4)})

    print("=" * 65)

    # Save CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"yolo_{opt.split}_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["class", "AP50", "AP50_95"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"class": "GLOBAL", "AP50": round(map50, 4), "AP50_95": round(map5095, 4)})

    # Save Markdown
    md_path = os.path.join(RESULTS_DIR, f"yolo_{opt.split}_metrics.md")
    with open(md_path, 'w') as f:
        f.write(f"# Metriche YOLO - Split: {opt.split}\n\n")
        f.write("| Classe | AP@0.5 | AP@0.5:0.95 |\n")
        f.write("|:---|:---:|:---:|\n")
        for row in rows:
            f.write(f"| {row['class']} | {row['AP50']*100:.2f}% | {row['AP50_95']*100:.2f}% |\n")
        f.write(f"| **GLOBAL** | **{map50*100:.2f}%** | **{map5095*100:.2f}%** |\n")

    print(f"\nResults saved to {csv_path} and {md_path}")

if __name__ == "__main__":
    main(parse_opt())
