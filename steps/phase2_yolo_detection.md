# Phase 2 — Data Exploration & YOLO Object Detection

## Obiettivo (dal brief)

> "Implement a pretrained object detection model (YOLO). The model will be used to detect and
> localize objects within images, with results being visualized using detected bounding boxes.
> Additionally, an evaluation module will be implemented to assess the performance of the
> detection model using mAP metric."
> — project6.md, Phase 2

---

## Dataset: LineMod Preprocessed

**Struttura attesa:**
```
datasets/linemod/Linemod_preprocessed/
├── data/
│   └── {01..15}/
│       ├── gt.yml        # per ogni img_id: obj_id, obj_bb [x,y,w,h], cam_R_m2c, cam_t_m2c
│       ├── info.yml      # per ogni img_id: cam_K (3x3 flat), depth_scale
│       ├── rgb/          # immagini RGB (640×480 PNG)
│       └── depth/        # depth maps (PNG 16-bit, valori in unità raw da scalare)
└── models/
    ├── obj_{01..15}.ply  # mesh 3D in millimetri
    └── models_info.yml   # per ogni obj_id: diameter, min_x/y/z, size_x/y/z
```

**15 oggetti** (class_id YOLO = obj_id − 1):

| obj_id | Nome        | class_id YOLO |
|--------|-------------|---------------|
| 1      | Ape         | 0             |
| 2      | Benchvise   | 1             |
| 3      | Bowl        | 2             |
| 4      | Cam         | 3             |
| 5      | Can         | 4             |
| 6      | Cat         | 5             |
| 7      | Cup         | 6             |
| 8      | Driller     | 7             |
| 9      | Duck        | 8             |
| 10     | Eggbox      | 9             |
| 11     | Glue        | 10            |
| 12     | Holepuncher | 11            |
| 13     | Iron        | 12            |
| 14     | Lamp        | 13            |
| 15     | Phone       | 14            |

**Annotazioni bbox:** nel formato `gt.yml`, ogni immagine ha una lista di annotazioni con campo `obj_bb: [x, y, w, h]` in pixel (angolo top-left + width/height).

**Depth:** le immagini di profondità sono PNG a 16-bit. Il valore reale in metri si ottiene con:
```
depth_meters = pixel_value * depth_scale / 1000.0
```
dove `depth_scale` è letto da `info.yml` per ogni immagine. I pixel con valore 0 rappresentano letture invalide del sensore.

---

## Split train / val / test

Generato con `data/split.py`, seed fisso `random_seed=42`, proporzioni **60% / 20% / 20%**.

Lo split è identico per tutti gli script del progetto (baseline, RGB-D, YOLO): stessa chiamata a `prepare_data_and_splits(ROOT_DATASET)` garantisce che train/val/test non si sovrappongano tra i diversi modelli.

---

## Modello YOLO

**Architettura:** YOLO11n (Ultralytics YOLO11, variante nano), punto di partenza `yolo11n.pt` (ImageNet pretrained).

**Motivazione della scelta:** YOLO11n è il modello più leggero della famiglia YOLO11, adatto per un task di detection 2D su immagini di dimensioni contenute (640×480) con 15 classi. L'obiettivo qui è ottenere bbox accurate per croppare l'oggetto — non massimizzare la mAP di detection in sé. Un modello nano garantisce inferenza rapida durante l'evaluation della posa.

**Formato dataset YOLO** (generato da `data/prepare_yolo_data.py`):
```
datasets/linemod/linemod_yolo_format/
├── images/{train,val,test}/  # copie delle RGB
├── labels/{train,val,test}/  # un .txt per immagine
│   └── obj{id:02d}_{img_id:04d}.txt  → "class_id cx cy w h" normalizzati [0,1]
└── data.yaml                 # nc=15, path ai tre split
```

**Hyperparameters di training** (`train_yolo.py`):

| Parametro  | Valore   |
|------------|----------|
| epochs     | 50       |
| batch      | 32       |
| imgsz      | 640      |
| optimizer  | auto     |
| pretrained | True     |
| device     | CUDA 0   |

---

## Training

```bash
python train_yolo.py
# oppure con override:
python train_yolo.py --epochs 100 --batch 16 --device cpu
```

I pesi vengono salvati da Ultralytics in `runs/detect/linemod_yolo_run/weights/best.pt`.
I pesi pre-allenati sono scaricabili via `download_data_and_weights.sh` in `weights/yolo/best.pt`.

---

## Evaluation — mAP (metrica del brief)

**Metrica standard per object detection:** mean Average Precision.

- **AP@0.5**: area sotto la curva precision-recall con soglia IoU = 0.5.
- **AP@0.5:0.95**: media di AP calcolata su soglie IoU da 0.5 a 0.95 con step 0.05. Metrica più stringente, standard COCO.
- **mAP**: media delle AP su tutte le classi.

### Script: `evaluate_yolo.py`

Valuta il modello pre-allenato sul **test split** (disgiunto da train e val) e produce un report per-classe.

```bash
python evaluate_yolo.py
# con opzioni:
python evaluate_yolo.py --split val --device cpu
```

Output:
```
=================================================================
YOLO Evaluation — split: test
=================================================================
Global mAP@0.5              XX.XX%
Global mAP@0.5:0.95         XX.XX%
Mean Precision              XX.XX%
Mean Recall                 XX.XX%

-----------------------------------------------------------------
Class           AP@0.5    AP@0.5:0.95
-----------------------------------------------------------------
Ape              XX.XX%        XX.XX%
Benchvise        XX.XX%        XX.XX%
...
=================================================================
Results saved to results/yolo_test_metrics.csv
```

Il CSV `results/yolo_test_metrics.csv` contiene una riga per classe + una riga `GLOBAL`.

**Prerequisito:** il dataset YOLO deve essere già preparato:
```bash
python data/prepare_yolo_data.py
```

---

## Visualizzazione (già implementata)

La visualizzazione delle bbox predette da YOLO è integrata negli script di visualizzazione della posa:

- `visualize_resultBaseline.py` — disegna in **blu** la bbox YOLO, in verde la pose GT, in rosso la pose predetta.
- `visualize_resultRGBD.py` — identico, con in più le metriche ADD/dT/dR a schermo.

Entrambi mostrano la bbox YOLO come elemento di ispezione del primo stadio della pipeline.

---

## Note tecniche rilevanti

### Augmentation YOLO vs pose estimation
Ultralytics applica di default augmentation aggressiva durante il training YOLO (mosaic, hflip con `fliplr=0.5`, HSV jitter). Per la detection 2D questo è standard e aumenta la robustezza. Tuttavia, il flip orizzontale **cambia la posa 3D dell'oggetto** — il che sarebbe un problema per un task di pose estimation diretta. Qui il YOLO fa solo detection 2D (bbox), quindi il flip non crea inconsistenze.

### Ruolo del YOLO nella pipeline di posa
Il YOLO non è valutato come sistema standalone: il suo output (bbox) alimenta il pose estimator. Quindi due scenari di errore sono distinti:
1. **YOLO miss**: oggetto non detectato → sample saltato nell'evaluation della posa.
2. **YOLO bbox inaccurata**: crop rumoroso → degrada il pose estimator. L'evaluation separata del baseline con GT crop vs YOLO crop (in `evaluate_metricsBaseline.py`) misura esattamente questo impatto.

### Mapping class_id
`class_id_yolo = obj_id - 1` (LineMod usa IDs da 1, YOLO da 0).
Tutti gli script che interpretano le predizioni YOLO usano `int(box.cls) != target_cls` dove `target_cls = obj_id - 1` (es. `utils/rgbd_utils.py:100-109`, `select_detection_for_object`).
