"""Does the single anchor degrade under occlusion?

The paper lists the single-anchor design as a limitation under heavy occlusion.
That claim is a conjecture unless measured, so this script measures it on the
official test split.

Visibility proxy: project the CAD model points with the ground-truth pose and
count how many land inside the object mask. Points hidden by the object itself
still land inside its silhouette, so the ratio drops mainly under *external*
occlusion — which is what we care about.

For each sampled frame we record
  visibility   fraction of projected model points falling inside the mask
  anchor_err   distance between the depth-median anchor and the true center
  n_depth      number of valid depth pixels used to build the anchor
and join them with the per-sample ADD errors of the trained model.

Local, CPU only.  Usage:  python -m tools.occlusion_analysis [n_per_object]
"""
import os
import sys
import csv
from collections import defaultdict

import cv2
import numpy as np
import yaml
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data_split import prepare_data_and_splits
from phase4_fusion.main.dataset import LineModDatasetRGBD
from phase4_fusion.main.rgbd_utils import load_info_cache, fetch_sample_info, convert_depth_to_meters

ROOT = "datasets/linemod/Linemod_preprocessed"
OUT = os.path.expanduser("~/StudioAML/04_settembre/occlusion_analysis.csv")
PERSAMPLE = os.path.expanduser("~/StudioAML/04_settembre/persample_norm_s42.csv")
NAMES = {1: "ape", 2: "benchvise", 4: "camera", 5: "can", 6: "cat", 8: "driller", 9: "duck",
         10: "eggbox", 11: "glue", 12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"}


def main(n_per_obj=250):
    rng = np.random.RandomState(0)
    _, _, test_samples, gt_cache = prepare_data_and_splits(ROOT)
    info = load_info_cache(ROOT, sorted(gt_cache.keys()))
    models_info = yaml.safe_load(open(f"{ROOT}/models/models_info.yml"))

    model_err = {}
    if os.path.exists(PERSAMPLE):
        with open(PERSAMPLE) as f:
            for r in csv.DictReader(f):
                model_err[(int(r["obj_id"]), int(r["img_id"]))] = (float(r["err_mm"]), int(r["hit"]))
    print(f"per-sample errors loaded: {len(model_err)}")

    pts_cache = {}
    for oid in sorted(gt_cache.keys()):
        v = np.asarray(trimesh.load(f"{ROOT}/models/obj_{oid:02d}.ply").vertices, dtype=np.float64)
        pts_cache[oid] = (v[rng.choice(len(v), 800, replace=False)] if len(v) > 800 else v) / 1000.0

    by_obj = defaultdict(list)
    for oid, img_id in test_samples:
        by_obj[oid].append(img_id)

    rows = []
    for oid in sorted(by_obj):
        ids = by_obj[oid]
        pick = rng.choice(len(ids), min(n_per_obj, len(ids)), replace=False)
        for k in pick:
            img_id = ids[k]
            ann = next(a for a in gt_cache[oid][img_id] if a["obj_id"] == oid)
            R = np.array(ann["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            T = np.array(ann["cam_t_m2c"], dtype=np.float64) / 1000.0
            bbox = ann["obj_bb"]
            entry = fetch_sample_info(info, oid, img_id)
            K = np.array(entry["cam_K"], dtype=np.float64).reshape(3, 3)

            mask = cv2.imread(f"{ROOT}/data/{oid:02d}/mask/{img_id:04d}.png", cv2.IMREAD_GRAYSCALE)
            depth_raw = cv2.imread(f"{ROOT}/data/{oid:02d}/depth/{img_id:04d}.png", cv2.IMREAD_UNCHANGED)
            if mask is None or depth_raw is None:
                continue
            depth_m = convert_depth_to_meters(depth_raw, entry.get("depth_scale", 1.0))

            # visibility: projected model points landing inside the mask
            p_cam = pts_cache[oid] @ R.T + T
            z = np.clip(p_cam[:, 2], 1e-6, None)
            u = (p_cam[:, 0] * K[0, 0] / z + K[0, 2]).round().astype(int)
            v_ = (p_cam[:, 1] * K[1, 1] / z + K[1, 2]).round().astype(int)
            h, w = mask.shape
            ok = (u >= 0) & (u < w) & (v_ >= 0) & (v_ < h)
            inside = np.zeros(len(u), bool)
            inside[ok] = mask[v_[ok], u[ok]] > 0
            visibility = inside.mean()

            # anchor as the model computes it, vs the true object center
            anchor = LineModDatasetRGBD._bbox_anchor(depth_m, bbox, K.astype(np.float32))
            anchor_err = float(np.linalg.norm(anchor - T) * 1000.0)  # mm
            x, y, bw, bh = [int(t) for t in bbox]
            box_d = depth_m[max(0, y):y + bh, max(0, x):x + bw]
            n_depth = int((box_d > 0).sum())

            err, hit = model_err.get((oid, img_id), (np.nan, -1))
            rows.append((oid, NAMES.get(oid, oid), img_id, round(visibility, 4),
                         round(anchor_err, 2), n_depth, err, hit))
        print(f"  {NAMES.get(oid, oid):<12} done ({len(rows)} rows)", flush=True)

    with open(OUT, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["obj_id", "object", "img_id", "visibility", "anchor_err_mm", "n_depth_px",
                       "model_err_mm", "hit"])
        wcsv.writerows(rows)

    vis = np.array([r[3] for r in rows])
    aerr = np.array([r[4] for r in rows])
    merr = np.array([r[6] for r in rows], dtype=float)
    hit = np.array([r[7] for r in rows])
    valid = ~np.isnan(merr) & (hit >= 0)

    print("\n" + "=" * 70)
    print(f"campioni: {len(rows)}   con errore modello: {valid.sum()}")
    print(f"visibility: min {vis.min():.3f}  p5 {np.percentile(vis,5):.3f}  "
          f"mediana {np.median(vis):.3f}  max {vis.max():.3f}")
    print("-" * 70)
    print(f"{'bin visibility':>18}{'n':>7}{'anchor err mm':>15}{'model err mm':>14}{'ADD %':>9}")
    edges = [0, .5, .7, .8, .9, .95, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (vis >= lo) & (vis < hi)
        if m.sum() == 0:
            continue
        mv = m & valid
        print(f"{f'{lo:.2f}-{hi:.2f}':>18}{m.sum():>7}{aerr[m].mean():>15.1f}"
              f"{(np.nanmean(merr[mv]) if mv.sum() else float('nan')):>14.1f}"
              f"{(100*hit[mv].mean() if mv.sum() else float('nan')):>9.1f}")
    print("-" * 70)
    print(f"correlazione visibility vs anchor_err : {np.corrcoef(vis, aerr)[0,1]:+.3f}")
    if valid.sum() > 10:
        print(f"correlazione visibility vs model_err  : {np.corrcoef(vis[valid], merr[valid])[0,1]:+.3f}")
        print(f"correlazione anchor_err vs model_err  : {np.corrcoef(aerr[valid], merr[valid])[0,1]:+.3f}")
    print("=" * 70)
    print(f"csv -> {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250)
