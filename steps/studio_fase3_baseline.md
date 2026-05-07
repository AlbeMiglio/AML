# Studio Approfondito — Fase 3: Baseline RGB-only Pose Estimation

Questo documento analizza in dettaglio ogni file della Fase 3, spiegando il *cosa fa*, il *come funziona* e il *perché* di ogni scelta. L'obiettivo è darti una comprensione completa per poter spiegare il progetto nel report e all'orale.

---

## 1. Il Modello: `phase3_baseline/model.py`

### Cosa fa
Definisce la rete neurale `PosePredictor` che, data un'immagine RGB ritagliata su un oggetto, predice:
- **Rotazione** → quaternione unitario a 4 dimensioni (x, y, z, w)
- **Traslazione** → vettore 3D in metri (x, y, z)

### Come funziona (riga per riga)

```python
self.backbone = models.resnet50(weights='IMAGENET1K_V1')
```
Carica ResNet-50 con pesi pre-addestrati su ImageNet (1.2M immagini, 1000 classi). Questo backbone sa già estrarre feature visive generali (bordi, texture, forme).

```python
num_features = self.backbone.fc.in_features  # = 2048
self.backbone.fc = nn.Identity()
```
ResNet-50 originalmente ha un layer finale `fc` che classifica in 1000 classi ImageNet. Noi non vogliamo classificare, vogliamo regredire una posa → rimuoviamo quel layer sostituendolo con `Identity()` (non fa nulla, lascia passare il vettore di 2048 feature).

```python
self.regression_head = nn.Linear(num_features, 4)  # quaternione
self.tvec_head = nn.Linear(num_features, 3)         # traslazione
```
Due "teste" lineari separate:
- `regression_head`: 2048 → 4 (quaternione). Il nome `regression_head` è mantenuto per compatibilità con i vecchi checkpoint.
- `tvec_head`: 2048 → 3 (traslazione XYZ).

```python
def forward(self, x):
    features = self.backbone(x)                              # (B, 2048)
    quat = F.normalize(self.regression_head(features), p=2, dim=1)  # (B, 4)
    tvec = self.tvec_head(features)                          # (B, 3)
    return quat, tvec
```
`F.normalize` forza il quaternione ad avere norma 1 (quaternione unitario). Questo è fondamentale: un quaternione non normalizzato non rappresenta una rotazione valida.

### Perché queste scelte
- **ResNet-50**: backbone standard per transfer learning, bilancio tra potenza e velocità.
- **Due teste separate**: rotazione e traslazione sono grandezze diverse (angolare vs metrica), meglio predirle indipendentemente.
- **Normalizzazione nel forward** (non nella loss): garantisce che l'output sia SEMPRE un quaternione valido, anche durante l'inferenza.

### Schema visivo

```
Immagine RGB (224×224×3)
         ↓
   ResNet-50 backbone (senza ultimo layer)
         ↓
   Feature vector (2048D)
      ↙        ↘
regression_head   tvec_head
   (2048→4)       (2048→3)
      ↓              ↓
 F.normalize         ↓
      ↓              ↓
 Quaternione (4D)  Traslazione (3D)
```

---

## 2. Il Dataset: `phase3_baseline/dataset.py`

### Cosa fa
Carica le immagini dal dataset LineMod, le ritaglia attorno all'oggetto e le prepara per la rete.

### Flusso di ogni singolo campione (`__getitem__`)

**Step 1 — Caricamento immagine:**
```python
img = Image.open(img_path).convert("RGB")
```

**Step 2 — Lettura annotazione (bounding box + posa):**
```python
x, y, w, h = target_ann['obj_bb']   # bbox in pixel (top-left + width/height)
R_mat = target_ann['cam_R_m2c']      # matrice rotazione 3×3 (GT)
T = target_ann['cam_t_m2c']          # traslazione in mm (GT)
```

**Step 3 — Crop quadrato:**
```python
center_x = x + w / 2
center_y = y + h / 2
side = max(w, h)      # prende il lato più lungo come riferimento
```
Il crop è reso quadrato prendendo `max(w, h)` come lato. Perché? ResNet si aspetta immagini quadrate (224×224). Se croppassi rettangolare e poi ridimensionassi, deformeresti l'oggetto. Un crop quadrato mantiene le proporzioni.

**Step 4 — Resize e trasformazioni:**
```python
img_resized = img_crop.resize((224, 224), Image.BILINEAR)
img_tensor = self.transform(img_resized)
```
L'immagine viene portata a 224×224 pixel (standard ResNet) e normalizzata con media/std di ImageNet.

**Step 5 — Conversione rotazione → quaternione:**
```python
quaternion_gt = matrix_to_quaternion(R_mat)
```
La GT nel dataset è una matrice 3×3. La rete predice quaternioni → serve la conversione.

### Data Augmentation (solo in training)

| Augmentation | Parametri | Perché |
|---|---|---|
| `ColorJitter` | brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05 | Simula variazioni di illuminazione |
| `GaussianBlur` | kernel=3, sigma=(0.1, 1.5), p=0.2 | Simula sfocatura del sensore |
| `RandomErasing` | p=0.25, scale=(0.02, 0.1) | Simula occlusioni parziali |

**IMPORTANTE**: NON vengono usati flip orizzontali né rotazioni dell'immagine, perché questi cambierebbero la posa GT senza che il dataset lo sappia → la rete imparerebbe informazioni sbagliate.

---

## 3. Le Funzioni di Costo: `phase3_baseline/losses.py`

### Rotation Loss

```python
def rotation_loss(q_pred, q_true):
    inner_prod = torch.abs(torch.sum(q_pred * q_true, dim=1))
    return torch.mean(1 - inner_prod)
```

**Formula:** `L_rot = 1 − |q_pred · q_true|`

**Spiegazione intuitiva:**
- Il prodotto scalare tra due quaternioni unitari misura quanto sono "allineati".
- Se `q_pred = q_true` → prodotto scalare = 1 → loss = 0 (perfetto!)
- Se sono opposti → prodotto scalare = 0 → loss = 1 (massimo errore)

**Perché il valore assoluto `|...|`?**
I quaternioni hanno un'ambiguità di segno: `q` e `-q` rappresentano la STESSA rotazione (ruotare di θ attorno all'asse n equivale a ruotare di -θ attorno a -n). Senza `abs()`, la loss potrebbe penalizzare un quaternione perfettamente corretto ma con segno opposto.

**Range:** La loss è sempre tra 0 (perfetto) e 1 (rotazione a 180°).

### Translation Loss

```python
def translation_loss(t_pred, t_true):
    return F.mse_loss(t_pred, t_true)
```

**Formula:** `L_trans = MSE(t_pred, t_true) = media( (x_pred−x_gt)² + (y_pred−y_gt)² + (z_pred−z_gt)² )`

Semplice errore quadratico medio tra i vettori di traslazione. Entrambi sono in **metri** (la conversione da mm avviene nel training loop).

### Traslazione Pinhole (per l'evaluation)

```python
def compute_pinhole_translation(bbox, intrinsics, real_diameter):
    Z = (fx * real_diameter) / pixel_size    # stima della profondità
    X = (u_center - cx) * Z / fx             # proiezione inversa X
    Y = (v_center - cy) * Z / fy             # proiezione inversa Y
```

**Cosa fa:** Stima la posizione 3D dell'oggetto usando solo la geometria della camera (senza rete neurale).

**Come funziona:**
1. Dalla dimensione in pixel della bbox e dal diametro reale dell'oggetto (noto), stima la distanza Z.
2. Dal centro della bbox e dai parametri intrinseci della camera (focal length, punto principale), calcola X e Y.

**Perché esiste:** Serve come confronto nell'evaluation: "quanto è meglio/peggio la traslazione appresa dalla rete rispetto a una semplice stima geometrica?"

### Conversione matrice → quaternione

```python
def matrix_to_quaternion(matrix_3x3):
    r = R.from_matrix(matrix_3x3)
    return torch.tensor(r.as_quat(), dtype=torch.float32)  # (x, y, z, w)
```

Usa `scipy.spatial.transform.Rotation`. La convenzione di scipy è `(x, y, z, w)`, coerente con tutto il resto del codice.

---

## 4. Il Training: `phase3_baseline/train.py`

### Strategia di addestramento in due fasi

| Epoche | Backbone | Cosa impara | Perché |
|---|---|---|---|
| 1-10 | ❄️ Congelato | Solo le due teste (quat + tvec) | Protegge i pesi ImageNet dai gradienti iniziali forti e caotici |
| 11-50 | 🔓 Scongelato (LR×0.1) | Tutta la rete (fine-tuning) | Permette al backbone di adattarsi al dominio LineMod, ma lentamente |

Questa strategia si chiama **gradual unfreezing** ed è una best practice nel transfer learning.

### Loss combinata

```python
loss = rotation_loss(pred_quat, gt_quat) + LAMBDA_T * translation_loss(pred_tvec, gt_T_m)
```

- `LAMBDA_T = 1.0`: peso relativo tra le due loss.
- `gt_T_m = batch["T"] / 1000.0`: conversione mm → metri (il dataset fornisce la T in mm).
- Le due loss sono ragionevolmente bilanciate: `L_rot ∈ [0, 1]` e `L_trans` tipicamente ~0.01-0.1 m².

### Scheduler

```python
ReduceLROnPlateau(mode='min', factor=0.5, patience=5)
```

Se la validation loss non migliora per 5 epoche consecutive, il learning rate viene dimezzato. Aiuta a superare i plateau nell'addestramento.

### Metriche tracciate su WandB

| Metrica | Significato |
|---|---|
| `train/loss`, `val/loss` | Loss combinata (rotazione + traslazione) |
| `train/deg`, `val/deg` | Errore angolare medio in gradi |
| `train/t_mse`, `val/t_mse` | MSE sulla traslazione (m²) |
| `backbone_unfrozen` | Flag 0/1: segnala il momento dello scongelamento |

### Salvataggio modello
- **Checkpoint** (`pose_resnet50_baseline_checkpoint.pth`): salvato ad ogni epoca, contiene tutto (pesi, optimizer, scheduler, epoca) per poter riprendere il training in caso di interruzione.
- **Best model** (`pose_resnet50_baseline_best.pth`): salvato solo quando la val loss migliora. È questo il file usato per la valutazione finale.

---

## 5. La Valutazione: `phase3_baseline/evaluate.py`

### Le 4 modalità di valutazione

Lo script valuta il modello in 4 modi diversi per isolare le fonti di errore:

| # | Modalità | R (rotazione) da | T (traslazione) da | Cosa misura |
|---|---|---|---|---|
| 1 | **GT Crop** | Crop con bbox vera | T vera (GT) | Qualità pura della rotazione predetta |
| 2 | **YOLO + T_gt** | Crop con bbox YOLO | T vera (GT) | Impatto del rumore introdotto da YOLO sulla rotazione |
| 3 | **YOLO + T_pinhole** | Crop con bbox YOLO | Stima geometrica | Pipeline geometrica completa (senza learning sulla T) |
| 4 | **YOLO + T_pred** | Crop con bbox YOLO | Predetta dal modello | Pipeline completamente appresa |

### Come leggere i risultati

**Confronto colonna 1 vs 2:** Se i numeri peggiorano molto, significa che la bbox di YOLO è imprecisa e il crop rumoroso degrada la rotazione. Se restano simili, YOLO funziona bene (come nel nostro caso con 99.5% mAP).

**Confronto colonna 3 vs 4:** Risponde alla domanda "è meglio stimare la T geometricamente o impararla con la rete?"

### La metrica ADD

```
ADD = media_i( || (R_gt · p_i + T_gt) − (R_pred · p_i + T_pred) || )
```

**In parole semplici:**
1. Prendi 500 punti 3D del modello CAD dell'oggetto (dal file `.ply`).
2. Trasformali con la posa vera (R_gt, T_gt) → ottieni dove stanno davvero nello spazio.
3. Trasformali con la posa predetta (R_pred, T_pred) → ottieni dove il modello pensa che stiano.
4. Calcola la distanza media tra punti corrispondenti.

**Soglia di successo:** ADD < 10% del diametro dell'oggetto → il campione è considerato corretto.

### ADD-S per oggetti simmetrici (eggbox, glue)

**Problema:** L'eggbox ruotato di 180° sembra identico. Con ADD standard, una posa corretta ma speculare verrebbe penalizzata come completamente sbagliata.

**Soluzione (ADD-S):** Invece di confrontare punto-per-punto, per ogni punto predetto si cerca il punto GT più vicino:

```
ADD-S = media_i( min_j( || p_pred_i − p_gt_j || ) )
```

Questo è lo standard nella comunità scientifica (benchmark BOP) per `eggbox` (obj_id=10) e `glue` (obj_id=11).

---

## 6. Lo Split dei Dati: `common/data_split.py`

### Cosa fa
Divide tutte le immagini del dataset in tre insiemi disgiunti:
- **Training** (60%): usato per addestrare la rete
- **Validation** (20%): usato durante il training per monitorare overfitting
- **Test** (20%): usato SOLO alla fine per la valutazione finale

### Dettagli importanti

- **Seed fisso** (`random_seed=42`): lo split è deterministico. Chiunque esegua il codice ottiene gli stessi identici insiemi. Fondamentale per la riproducibilità.
- **Condiviso tra tutte le fasi**: YOLO (Fase 2), Baseline (Fase 3) e RGB-D (Fase 4) usano tutti la stessa funzione `prepare_data_and_splits`. Questo garantisce che non ci siano "contaminazioni" (un'immagine di test di una fase non appare nel training di un'altra).

---

## 7. Dominio Gap: Training vs Evaluation

### Il problema
In **training**, il modello riceve il crop dell'immagine basato sulla bounding box **vera** (dal file `gt.yml`). In **evaluation end-to-end**, il crop viene dalla bounding box **predetta da YOLO**, che può essere leggermente diversa (spostata, più grande, più piccola).

### Conseguenza
Il modello vede input leggermente diversi da quelli su cui è stato addestrato. Questo introduce un "dominio gap". Ecco perché le performance in valutazione sono tipicamente inferiori a quelle in training.

### Come lo misuriamo
Confrontando la colonna 1 (GT Crop) con la colonna 2 (YOLO Crop) nella tabella di evaluation: la differenza tra le due quantifica esattamente l'impatto di questo gap.

---

## 8. Riepilogo: Flusso Completo della Fase 3

```
                    TRAINING                              EVALUATION
                    --------                              ----------
Dataset LineMod                                    Dataset LineMod (test split)
      ↓                                                    ↓
Crop con GT bbox                                   Crop con YOLO bbox
      ↓                                                    ↓
Augmentation (ColorJitter, Blur, Erasing)           Nessuna augmentation
      ↓                                                    ↓
Resize 224×224 + normalizzazione ImageNet           Resize 224×224 + normalizzazione
      ↓                                                    ↓
        ┌──── PosePredictor (ResNet-50) ────┐
        │                                   │
        ↓                                   ↓
   Quaternione (4D)                  Traslazione (3D)
        ↓                                   ↓
   rotation_loss                    translation_loss
        ↓                                   ↓
        └────── Loss combinata ─────────────┘
                     ↓
              Backpropagation
                     ↓
            Aggiornamento pesi
```

### Per il report: punti chiave da menzionare

1. **Transfer Learning**: ResNet-50 pre-addestrato su ImageNet, fine-tuning graduale.
2. **Dual-head architecture**: due teste separate per rotazione (quaternione) e traslazione (vettore 3D).
3. **Combined Loss**: `L = L_rot + λ · L_trans`, dove L_rot usa il prodotto scalare tra quaternioni e L_trans è MSE.
4. **Evaluation a 4 modalità**: isola le fonti di errore (crop, rotazione, traslazione).
5. **ADD/ADD-S**: metrica standard per 6D pose estimation, con variante simmetrica per eggbox e glue.
6. **Dominio gap**: differenza tra GT bbox (training) e YOLO bbox (evaluation) impatta le performance.
