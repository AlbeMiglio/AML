# Studio Approfondito — Fase 2: Data Exploration e YOLO Object Detection

Questo documento analizza in dettaglio ogni file della Fase 2, spiegando il *cosa fa*, il *come funziona* e il *perché* di ogni scelta.

---

## 1. Panoramica: Cosa fa la Fase 2

La Fase 2 ha tre obiettivi:
1. **Esplorare il dataset** LineMod (capire la struttura, le annotazioni, i formati)
2. **Addestrare YOLO** per riconoscere e localizzare i 15 oggetti con bounding box
3. **Valutare YOLO** usando la metrica mAP per misurare la qualità della detection

YOLO non è il fine ultimo del progetto: il suo ruolo è produrre le bounding box che verranno usate nelle Fasi 3 e 4 per croppare le immagini prima di stimare la posa 6D.

---

## 2. Preparazione Dati: `phase2_detection/prepare_yolo_data.py`

### Cosa fa
Converte il dataset LineMod dal suo formato originale al formato che YOLO si aspetta.

### Il problema
LineMod ha le annotazioni in file `gt.yml` con bounding box in formato `[x, y, w, h]` (angolo top-left + dimensioni in pixel). YOLO invece vuole un file `.txt` per ogni immagine con coordinate **normalizzate del centro**:
```
class_id  center_x  center_y  width  height
```
dove tutti i valori sono tra 0 e 1 (divisi per la dimensione dell'immagine).

### La conversione (righe 155-161)

```python
# Da formato LineMod (top-left + dimensioni pixel)
x_min, y_min, w, h = raw_bb

# A formato YOLO (centro normalizzato)
cx_norm = (x_min + w / 2) / img_w    # centro X normalizzato
cy_norm = (y_min + h / 2) / img_h    # centro Y normalizzato
w_norm  = w / img_w                   # larghezza normalizzata
h_norm  = h / img_h                   # altezza normalizzata
```

**Esempio numerico:** Se l'immagine è 640×480 e la bbox è `[100, 200, 60, 80]`:
- Centro X: (100 + 30) / 640 = 0.203
- Centro Y: (200 + 40) / 480 = 0.500
- Larghezza: 60 / 640 = 0.094
- Altezza: 80 / 480 = 0.167
- File `.txt`: `3 0.203125 0.500000 0.093750 0.166667`

### Mapping delle classi (riga 161)

```python
class_id = obj_id - 1
```

LineMod numera gli oggetti da **1** a 15. YOLO numera le classi da **0** a 14. Senza questa conversione, YOLO cercherebbe una classe 15 che non esiste.

| obj_id (LineMod) | class_id (YOLO) | Nome |
|---|---|---|
| 1 | 0 | Ape |
| 2 | 1 | Benchvise |
| ... | ... | ... |
| 15 | 14 | Phone |

### Split dei dati (riga 105)

```python
train_samples, val_samples, test_samples, _ = prepare_data_and_splits(SOURCE_ROOT)
```

Usa la **stessa funzione** di split condivisa con le Fasi 3 e 4 (`common/data_split.py`), con seed=42 e proporzioni 60/20/20. Questo garantisce che le stesse immagini siano nello stesso split in tutte le fasi.

### Il file `data.yaml` (righe 174-185)

```yaml
path: datasets/linemod/linemod_yolo_format
train: images/train
val: images/val
test: images/test
nc: 15
names: {0: Ape, 1: Benchvise, ...}
```

È il file di configurazione che dice a YOLO: dove trovare le immagini, quante classi ci sono e come si chiamano. Sia il training che l'evaluation lo leggono.

### Struttura output

```
datasets/linemod/linemod_yolo_format/
├── images/
│   ├── train/   → 9480 immagini PNG
│   ├── val/     → 3160 immagini PNG
│   └── test/    → 3160 immagini PNG
├── labels/
│   ├── train/   → 9480 file .txt (un'etichetta per immagine)
│   ├── val/     → 3160 file .txt
│   └── test/    → 3160 file .txt
└── data.yaml    → configurazione
```

---

## 3. Il Training: `phase2_detection/train.py`

### Cosa fa
Addestra il modello YOLO11n (la variante nano di YOLO versione 11) sul dataset LineMod preparato.

### Transfer Learning (concetto chiave)

```python
model = YOLO('yolo11n.pt')   # pesi pre-addestrati su COCO/ImageNet
```

Il file `yolo11n.pt` contiene un modello YOLO già addestrato su milioni di immagini generiche. Sa già riconoscere bordi, forme e oggetti comuni (persone, auto, cani...). Il training lo specializza a riconoscere i 15 oggetti LineMod.

**Senza transfer learning:** la rete partirebbe da pesi casuali e ci vorrebbero molte più epoche per convergere.
**Con transfer learning:** il backbone sa già "vedere" → impara i nuovi oggetti molto più velocemente.

### Hyperparameters

| Parametro | Valore | Significato |
|---|---|---|
| `epochs` | 50 | Numero di passaggi sull'intero dataset |
| `batch` | 32 | Immagini processate in parallelo |
| `imgsz` | 640 | Le immagini vengono ridimensionate a 640px |
| `optimizer` | auto | Ultralytics sceglie il migliore (tipicamente SGD o AdamW) |
| `pretrained` | True | Parte da `yolo11n.pt`, non da zero |

### Data Augmentation di YOLO

Ultralytics applica automaticamente augmentation aggressive durante il training:
- **Mosaic**: combina 4 immagini in una sola (aumenta la varietà di scene)
- **Horizontal flip** (`fliplr=0.5`): specchia orizzontalmente con probabilità 50%
- **HSV jitter**: varia tonalità, saturazione e luminosità

**Nota tecnica importante:** Il flip orizzontale cambia la posa 3D reale dell'oggetto. Ma YOLO fa solo detection 2D (trova il rettangolo), non stima la posa → il flip non crea problemi. Se YOLO stimasse anche la posa, il flip sarebbe un errore.

### WandB (Weights & Biases)

```python
wandb.init(project=opt.wandb_project, name=opt.name, config=vars(opt))
```

Il training logga le metriche su WandB, una piattaforma online per monitorare gli esperimenti di ML. Permette di vedere grafici in tempo reale di loss, mAP, precision, recall durante il training.

### Output del training

Il training produce automaticamente:
```
runs/detect/linemod_yolo_run/
├── weights/
│   ├── best.pt      ← modello migliore (quello che usiamo)
│   └── last.pt      ← ultimo modello salvato
├── results.csv      ← metriche per ogni epoca
└── ...              ← grafici, confusion matrix, ecc.
```

`best.pt` è selezionato automaticamente come il modello con la miglior mAP sul validation set.

---

## 4. La Valutazione: `phase2_detection/evaluate.py`

### Cosa fa
Carica il modello YOLO addestrato (`best.pt`) e ne misura le performance sul **test split** (immagini mai viste durante il training).

### Metriche calcolate

#### Precision (Precisione)
```
Precision = Veri Positivi / (Veri Positivi + Falsi Positivi)
```
"Di tutte le detection che YOLO ha fatto, quante erano corrette?" Un falso positivo è quando YOLO dice "qui c'è un oggetto" ma in realtà non c'è.

#### Recall (Richiamo)
```
Recall = Veri Positivi / (Veri Positivi + Falsi Negativi)
```
"Di tutti gli oggetti realmente presenti, quanti li ha trovati?" Un falso negativo è quando l'oggetto c'è ma YOLO non lo vede.

#### AP (Average Precision)
L'area sotto la curva Precision-Recall. Calcolata per ogni singola classe.

#### mAP (mean Average Precision)
La media delle AP su tutte le classi. È LA metrica principale per valutare un detector.

### Due varianti di mAP

| Metrica | Soglia IoU | Significato |
|---|---|---|
| **mAP@0.5** | IoU ≥ 0.5 | La bbox predetta deve sovrapporsi almeno al 50% con quella vera |
| **mAP@0.5:0.95** | Media su IoU da 0.5 a 0.95 | Molto più stringente, standard COCO |

### IoU (Intersection over Union)

```
IoU = Area di intersezione / Area di unione
```

Misura quanto si sovrappongono la bbox predetta e quella reale:
- IoU = 1.0 → perfettamente sovrapposte
- IoU = 0.5 → sovrapposizione parziale (soglia minima per considerare la detection corretta)
- IoU = 0.0 → nessuna sovrapposizione

### Come funziona il codice (righe 37-43)

```python
metrics = model.val(
    data=opt.data,       # data.yaml con percorsi delle immagini
    split=opt.split,     # 'test' (default)
    imgsz=opt.imgsz,     # 640 pixel
    device=opt.device,   # GPU o CPU
    verbose=False,
)
```

`model.val()` è il metodo di Ultralytics che:
1. Carica tutte le immagini del split selezionato
2. Per ogni immagine, YOLO predice le bounding box
3. Confronta le predizioni con le annotazioni vere (ground truth)
4. Calcola tutte le metriche (mAP, precision, recall, AP per classe)

### Output

Lo script produce:
- **Stampa a terminale**: tabella con mAP globale e AP per ogni classe
- **File CSV** (`results/yolo_test_metrics.csv`): dati in formato tabellare per elaborazioni
- **File Markdown** (`results/yolo_test_metrics.md`): tabella leggibile per il report

---

## 5. L'architettura YOLO11n

### Perché YOLO11n (nano)?

YOLO11 è l'ultima generazione della famiglia YOLO di Ultralytics. La variante "n" (nano) è la più leggera:

| Caratteristica | Valore |
|---|---|
| Parametri | 2.6M |
| GFLOPs | 6.3 |
| Layers (fused) | 100 |

**Motivazione della scelta:** L'obiettivo non è massimizzare la mAP di detection (abbiamo già 99.5%!), ma avere un detector veloce per croppare le immagini nelle fasi successive. Un modello nano garantisce inferenza rapida durante l'evaluation della posa (Fasi 3-4), dove YOLO viene chiamato per ogni immagine.

### Come funziona YOLO (concetti generali)

1. **Input:** Immagine ridimensionata a 640×640 pixel
2. **Backbone:** Estrae feature multi-scala (a diverse risoluzioni)
3. **Neck:** Fonde le feature di scale diverse (Feature Pyramid Network)
4. **Head:** Per ogni posizione nella griglia, predice:
   - Probabilità che ci sia un oggetto
   - Coordinate della bounding box (x, y, w, h)
   - Probabilità per ogni classe
5. **Post-processing (NMS):** Elimina le detection duplicate/sovrapposte

YOLO è un detector **single-stage**: fa tutto in un unico passaggio (a differenza di Faster R-CNN che ha due fasi). Questo lo rende molto veloce.

---

## 6. Ruolo di YOLO nella Pipeline Complessiva

YOLO non è fine a sé stesso. Il suo output (bounding box) alimenta le fasi successive:

```
Immagine RGB (640×480)
       ↓
    YOLO11n
       ↓
  Bounding box [x1, y1, x2, y2]
       ↓
    Crop quadrato dell'oggetto
       ↓
  Resize 224×224
       ↓
  PosePredictor (Fase 3) o FusionPredictor (Fase 4)
       ↓
  Posa 6D (Rotazione + Traslazione)
```

### Due scenari di errore

1. **YOLO miss** (oggetto non trovato): il sample viene saltato nell'evaluation della posa → la metrica finale è calcolata solo sugli oggetti trovati
2. **YOLO bbox imprecisa**: il crop è "rumoroso" (tagliato male, troppo grande/piccolo) → il pose estimator riceve un input di qualità inferiore → la posa predetta peggiora

L'evaluation della Fase 3 confronta esplicitamente questi casi: la colonna "GT Crop" vs "YOLO Crop" nella tabella dei risultati misura esattamente l'impatto della qualità del detector.

---

## 7. Il Dataset LineMod

### Struttura

```
datasets/linemod/Linemod_preprocessed/
├── data/
│   └── {01..15}/          ← una cartella per ogni oggetto
│       ├── gt.yml         ← annotazioni (bbox, rotazione, traslazione)
│       ├── info.yml       ← parametri della camera (K, depth_scale)
│       ├── rgb/           ← immagini RGB (640×480 PNG)
│       └── depth/         ← mappe di profondità (PNG 16-bit)
└── models/
    ├── obj_{01..15}.ply   ← modelli 3D (mesh) in millimetri
    └── models_info.yml    ← diametro, dimensioni, bounding box 3D
```

### Formato delle annotazioni (`gt.yml`)

Per ogni immagine, il file contiene una lista di annotazioni:
```yaml
0:                              # img_id
  - obj_id: 1                  # quale oggetto
    obj_bb: [167, 82, 142, 164] # bbox [x, y, w, h] in pixel
    cam_R_m2c: [...]            # matrice di rotazione 3×3 (flat, 9 numeri)
    cam_t_m2c: [...]            # vettore di traslazione [X, Y, Z] in mm
```

### 15 oggetti del dataset

| ID | Oggetto | Note |
|---|---|---|
| 1 | Ape | Piccolo, textureless |
| 2 | Benchvise | Grande, strutturato |
| 4 | Camera | Medio |
| 5 | Can | Cilindrico |
| 6 | Cat | Piccolo, complesso |
| 8 | Driller | Grande, asimmetrico |
| 9 | Duck | Piccolo |
| 10 | Eggbox | **Simmetrico** (ADD-S) |
| 11 | Glue | **Simmetrico** (ADD-S) |
| 12 | Holepuncher | Medio |
| 13 | Iron | Grande |
| 14 | Lamp | Alto, sottile |
| 15 | Phone | Piatto |

Nota: gli ID 3 (Bowl) e 7 (Cup) non hanno dati nel test split usato dall'evaluation.

---

## 8. Riepilogo: Per il Report

### Punti chiave da menzionare nella Fase 2

1. **Dataset LineMod**: 15800 immagini, 15 oggetti, split 60/20/20 con seed fisso per riproducibilità
2. **Conversione formato**: da bbox `[x, y, w, h]` pixel (LineMod) a coordinate normalizzate del centro (YOLO)
3. **Transfer Learning**: YOLO11n parte da pesi pre-addestrati su ImageNet/COCO (`yolo11n.pt`) → fine-tuning su LineMod
4. **Architettura YOLO11n**: modello nano (2.6M parametri), scelto per velocità nell'inferenza (verrà usato in cascata nelle fasi successive)
5. **Metrica mAP**: mean Average Precision, calcolata su test set mai visto durante il training
6. **Ruolo nella pipeline**: YOLO produce le bbox che alimentano il pose estimator delle Fasi 3 e 4; non è fine a sé stesso
