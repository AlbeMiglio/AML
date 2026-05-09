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

Generato con `common/data_split.py`, seed fisso `random_seed=42`, proporzioni **60% / 20% / 20%**.

Lo split è identico per tutti gli script del progetto (baseline, RGB-D, YOLO): stessa chiamata a `prepare_data_and_splits(ROOT_DATASET)` garantisce che train/val/test non si sovrappongano tra i diversi modelli.

---

## Modello YOLO

**Architettura:** YOLO11n (Ultralytics YOLO11, variante nano), punto di partenza `yolo11n.pt` (ImageNet pretrained).

**Motivazione della scelta:** YOLO11n è il modello più leggero della famiglia YOLO11, adatto per un task di detection 2D su immagini di dimensioni contenute (640×480) con 15 classi. L'obiettivo qui è ottenere bbox accurate per croppare l'oggetto — non massimizzare la mAP di detection in sé. Un modello nano garantisce inferenza rapida durante l'evaluation della posa.

**Formato dataset YOLO** (generato da `phase2_detection/prepare_yolo_data.py`):
```
datasets/linemod/linemod_yolo_format/
├── images/{train,val,test}/  # copie delle RGB
├── labels/{train,val,test}/  # un .txt per immagine
│   └── obj{id:02d}_{img_id:04d}.txt  → "class_id cx cy w h" normalizzati [0,1]
└── data.yaml                 # nc=15, path ai tre split
```

**Hyperparameters di training** (`phase2_detection/train.py`):

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
python -m phase2_detection.train
# oppure con override:
python -m phase2_detection.train --epochs 100 --batch 16 --device cpu
```

I pesi vengono salvati da Ultralytics in `runs/detect/linemod_yolo_run/weights/best.pt`.
I pesi pre-allenati sono scaricabili via `download_data_and_weights.sh` in `weights/yolo/best.pt`.

---

## Evaluation — mAP (metrica del brief)

**Metrica standard per object detection:** mean Average Precision.

- **AP@0.5**: area sotto la curva precision-recall con soglia IoU = 0.5.
- **AP@0.5:0.95**: media di AP calcolata su soglie IoU da 0.5 a 0.95 con step 0.05. Metrica più stringente, standard COCO.
- **mAP**: media delle AP su tutte le classi.

### Script: `phase2_detection/evaluate.py`

Valuta il modello pre-allenato sul **test split** (disgiunto da train e val) e produce un report per-classe.

```bash
python -m phase2_detection.evaluate
# con opzioni:
python -m phase2_detection.evaluate --split val --device cpu
```

### Risultati Ottenuti (Test Split)

Dopo l'esecuzione della valutazione sul test split di LineMod (3160 immagini), i risultati ottenuti con il modello `best.pt` sono i seguenti:

=================================================================
YOLO Evaluation — split: test
=================================================================
Global mAP@0.5              99.50%
Global mAP@0.5:0.95         96.03%
Mean Precision              99.96%
Mean Recall                 100.00%

-----------------------------------------------------------------
Class               AP@0.5    AP@0.5:0.95
-----------------------------------------------------------------
Ape                 99.50%         93.78%
Benchvise           99.50%         97.04%
Cam                 99.50%         94.97%
Can                 99.50%         98.47%
Cat                 99.50%         95.91%
Driller             99.50%         96.20%
Duck                99.50%         95.00%
Eggbox              99.50%         97.51%
Glue                99.50%         93.65%
Holepuncher         99.50%         95.67%
Iron                99.50%         96.08%
Lamp                99.50%         97.40%
Phone               99.50%         96.74%
=================================================================
Results saved to results/yolo_test_metrics.csv e results/yolo_test_metrics.md

**Prerequisito:** il dataset YOLO deve essere già preparato:
```bash
python -m phase2_detection.prepare_yolo_data
```

---

## Visualizzazione (già implementata)

La visualizzazione delle bbox predette da YOLO è integrata negli script di visualizzazione della posa:

- `phase3_baseline/visualize.py` — disegna in **blu** la bbox YOLO, in verde la pose GT, in rosso la pose predetta.
- `phase4_fusion/main/visualize.py` — identico, con in più le metriche ADD/dT/dR a schermo.

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
### Note sull'ambiente e l'implementazione
- **Gestione Dipendenze**: L'ambiente iniziale era privo di `pip`. È stato necessario inizializzarlo tramite `python3 -m ensurepip` e installare `scikit-learn` separatamente per permettere il funzionamento degli script di split dei dati.
- **Dataset Linking**: Per evitare di duplicare GB di dati, è stato creato un link simbolico (`ln -s`) tra la cartella sulla Scrivania e la directory `datasets/` del progetto.
- **Weights Fix**: Se il file `best.pt` viene caricato come cartella (estratto), la libreria Ultralytics fallisce. È stato necessario ricompattare i contenuti in un archivio Zip con estensione `.pt` per renderlo caricabile.
