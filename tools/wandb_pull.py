#!/usr/bin/env python3
"""
Pull i .pth dei modelli Phase 4 (Main + Extension) caricati con `wandb.save()`
dalla WandB cloud al filesystem locale.

Phase 3 baseline NON ha `wandb.save` nel training, quindi non è recuperabile
da qui (deve essere copiato dal pod/PC fisso dove è stato addestrato).

Uso:
    export WANDB_API_KEY=...               # da https://wandb.ai/authorize
    python tools/wandb_pull.py             # cerca le ultime run "best"
    python tools/wandb_pull.py --entity albemiglio        # specifica entity
    python tools/wandb_pull.py --run-id <id>              # forza una run specifica

I file vengono salvati nei path coerenti con `evaluate.py`:
    results_4_main/pose_rgbd_fusion_best.pth
    results_4_ext/pose_rgbd_custom_1ch_best.pth
"""

import argparse
import os
import sys
from pathlib import Path


PROJECT = "linemod-pose-estimation"
TARGETS = [
    # (run_name_substring, expected_file_basename, local_destination)
    ("RGBD_augmentation",       "pose_rgbd_fusion_best.pth",      "results_4_main/pose_rgbd_fusion_best.pth"),
    ("Custom ResNet-10",         "pose_rgbd_custom_1ch_best.pth",  "results_4_ext/pose_rgbd_custom_1ch_best.pth"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=None,
                    help="WandB entity (username/team). Default: tuo account.")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--run-id", default=None,
                    help="Forza una specifica run-id (skip ricerca per nome).")
    ap.add_argument("--list", action="store_true",
                    help="Solo lista le run del progetto, non scarica nulla.")
    args = ap.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        print("❌ WANDB_API_KEY non settata. Recuperala da https://wandb.ai/authorize")
        print("   poi:  export WANDB_API_KEY=...")
        sys.exit(1)

    try:
        import wandb
    except ImportError:
        print("Installing wandb...")
        os.system(f"{sys.executable} -m pip install --quiet wandb")
        import wandb

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project

    try:
        runs = api.runs(project_path, order="-created_at")
    except Exception as e:
        # Probabile: entity sbagliato. Suggerisci come trovarla.
        print(f"❌ Impossibile leggere il progetto '{project_path}': {e}")
        print("   Verifica entity con:  wandb whoami")
        print("   Oppure passa --entity <tuo_username>")
        sys.exit(2)

    if args.list:
        print(f"Run nel progetto {project_path}:")
        for r in runs:
            print(f"  [{r.state}] {r.id}  '{r.name}'  ({r.created_at})")
        return

    runs = list(runs)
    if not runs:
        print(f"❌ Nessuna run trovata nel progetto {project_path}.")
        sys.exit(3)

    for name_substr, filename, dest in TARGETS:
        print(f"\n▶ Cerco run con nome contenente '{name_substr}'...")
        if args.run_id:
            run = api.run(f"{project_path}/{args.run_id}")
        else:
            matching = [r for r in runs if name_substr.lower() in r.name.lower()]
            if not matching:
                print(f"   ⚠️  Nessuna run trovata. Skip {filename}.")
                continue
            run = matching[0]   # più recente (runs è già ordinato desc)
            print(f"   ✓ Run scelta: '{run.name}' ({run.id}, {run.state})")

        # Cerca il file fra quelli associati alla run
        files = list(run.files())
        match = next((f for f in files if f.name.endswith(filename)), None)
        if match is None:
            print(f"   ⚠️  File '{filename}' non trovato nei files della run.")
            print(f"      File disponibili: {[f.name for f in files][:20]}")
            continue

        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists():
            print(f"   ℹ️  {dest_path} già presente, skip download.")
            continue

        print(f"   ⬇️  Scaricando {match.name} ({match.size/1e6:.1f} MB) → {dest_path}")
        match.download(root=str(dest_path.parent), replace=False)
        # wandb scarica con il nome originale; rinomino se serve
        downloaded = dest_path.parent / Path(match.name).name
        if downloaded != dest_path and downloaded.exists():
            downloaded.rename(dest_path)
        print(f"   ✓ Salvato in {dest_path}")

    print("\n✅ Done. Pesi disponibili in weights/")


if __name__ == "__main__":
    main()
