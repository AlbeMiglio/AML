import os

import yaml
from sklearn.model_selection import train_test_split


def prepare_data_and_splits(
    dataset_root,
    random_seed=42,
    val_ratio=0.2,
    test_ratio=0.2,
    **legacy_kwargs,
):
    """Return train/val/test splits plus cached ground-truth annotations."""
    if "test_size" in legacy_kwargs:
        val_ratio = legacy_kwargs["test_size"]

    holdout_ratio = val_ratio + test_ratio
    if not 0 < holdout_ratio < 1:
        raise ValueError("val_ratio + test_ratio must be between 0 and 1")

    all_samples = []
    gt_cache = {}

    data_path = os.path.join(dataset_root, "data")
    obj_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

    print(f"Scanning {len(obj_dirs)} objects...")

    for obj_dir in obj_dirs:
        try:
            obj_id = int(obj_dir)
        except ValueError:
            continue

        gt_file = os.path.join(data_path, obj_dir, "gt.yml")
        if not os.path.exists(gt_file):
            continue

        with open(gt_file, "r", encoding="utf-8") as f:
            gt_content = yaml.safe_load(f)
            gt_cache[obj_id] = gt_content

        for img_id in gt_content.keys():
            all_samples.append((obj_id, int(img_id)))

    train_samples, holdout_samples = train_test_split(
        all_samples,
        test_size=holdout_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    relative_test_ratio = test_ratio / holdout_ratio
    val_samples, test_samples = train_test_split(
        holdout_samples,
        test_size=relative_test_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    train_pct = (len(train_samples) / len(all_samples)) * 100
    val_pct = (len(val_samples) / len(all_samples)) * 100
    test_pct = (len(test_samples) / len(all_samples)) * 100

    print("Split completed:")
    print(f"   - Train: {len(train_samples)} samples ({train_pct:.1f}%)")
    print(f"   - Val:   {len(val_samples)} samples ({val_pct:.1f}%)")
    print(f"   - Test:  {len(test_samples)} samples ({test_pct:.1f}%)")

    return train_samples, val_samples, test_samples, gt_cache
