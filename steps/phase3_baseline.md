# Phase 3 — Baseline RGB-only Pose Estimation

## Obiettivo (dal brief)

> "The Pose Predictor Model is a CNN designed for 3D object pose estimation, focusing on
> predicting rotation in the form of quaternions. [...] The system trains using a loss function
> that combines translation loss and rotation loss."
> — project6.md, Phase 3

---

## Architettura: `PosePredictor` (`models/PosePredictor.py`)

**Backbone:** ResNet-50 con pesi ImageNet (`IMAGENET1K_V1`). Il layer `fc` originale viene sostituito con `nn.Identity()` per esporre il feature vector da 2048 dimensioni.

**Due teste di regressione:**

| Testa       | Output | Descrizione                              |
|-------------|--------|------------------------------------------|
| `quat_head` | 4D     | Quaternione unitario `(x, y, z, w)`      |
| `tvec_head` | 3D     | Vettore di traslazione in **metri**      |

**Normalizzazione nel forward:** il quaternione è normalizzato con `F.normalize(..., p=2, dim=1)` direttamente nell'output del modello, garantendo che ogni batch produca un quaternione unitario per definizione. La traslazione è predetta in scala libera.

```python
def forward(self, x):
    features = self.backbone(x)
    quat = F.normalize(self.quat_head(features), p=2, dim=1)
    tvec = self.tvec_head(features)
    return quat, tvec
```

### Compatibilità con pesi pre-esistenti
La testa quaternione mantiene il nome `regression_head` (invariato rispetto alla versione originale): i vecchi checkpoint caricano questa parte senza problemi. La nuova `tvec_head` non è presente nei vecchi pesi, quindi i `load_state_dict` usano `strict=False`: la chiave mancante viene ignorata e la testa parte da random-init. Nessun crash; la traslazione non è addestrata finché non si effettua un retraining con la nuova loss combinata.

---

## Input Pipeline

**Crop:** la `LineModDataset` (`data/linemod_dataset.py`) fa un crop quadrato centrato sul bounding box GT: `side = max(w, h)`, crop da `(center_x - side/2, center_y - side/2)` a `(center_x + side/2, center_y + side/2)`. Il crop esce dall'immagine se l'oggetto è vicino ai bordi — PIL gestisce automaticamente con zero-padding.

**Resize:** 224×224 bicubic.

**Normalizzazione:** media/std ImageNet `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`.

**Augmentation (solo training):**
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)`
- `GaussianBlur(kernel_size=3, sigma=(0.1,1.5))` con p=0.2
- `RandomErasing(p=0.25, scale=(0.02,0.1))` — simula occlusioni parziali

Nessun flip geometrico, nessuna rotazione dell'immagine: queste augmentation cambierebbero la posa GT 3D.

---

## Loss Function (`trainBaseline.py`)

La loss combina rotazione e traslazione:

```
Loss = rotation_loss(q_pred, q_gt) + λ_T · MSE(t_pred, t_gt)
```

### Rotation loss (`utils/resNetUtils.py:rotation_loss`)

```
L_rot = 1 - |q_pred · q_gt|
```

Basata sul prodotto scalare assoluto tra quaternioni normalizzati. Il valore assoluto rende la loss invariante all'ambiguità di segno del quaternione (−q e +q rappresentano la stessa rotazione). Valore in [0, 1]: 0 = rotazione perfetta, 1 = rotazione opposta.

### Translation loss (`utils/resNetUtils.py:translation_loss`)

```
L_trans = MSE(t_pred, t_gt)
```

MSE standard tra traslazioni in **metri**. Il dataset `linemod_dataset.py` restituisce `T` in mm (raw da `gt.yml`); la conversione a metri avviene in `trainBaseline.py` con `gt_T_m = batch["T"].to(DEVICE) / 1000.0`.

### Peso λ_T

`LAMBDA_T = 1.0` di default. Poiché `L_rot ∈ [0,1]` e `L_trans` è in m² (valori tipici LineMod: oggetti a ~0.3–0.7m, MSE ordine ~0.01–0.1 m²), i due termini sono ragionevolmente bilanciati. Se in training si osserva che `L_rot` domina o viceversa, si può tunearlo.

---

## Training (`trainBaseline.py`)

### Hyperparameters

| Parametro          | Valore |
|--------------------|--------|
| `BATCH_SIZE`       | 32     |
| `BASE_LR`          | 1e-4   |
| `BACKBONE_LR_FACTOR` | 0.1  |
| `WEIGHT_DECAY`     | 1e-4   |
| `EPOCHS`           | 50     |
| `FREEZE_EPOCHS`    | 10     |
| `LAMBDA_T`         | 1.0    |

### Strategia freeze/unfreeze backbone

Nelle prime 10 epoche (`FREEZE_EPOCHS`) i parametri del backbone ResNet-50 sono congelati: si ottimizzano solo le due teste di regressione. Questo evita di distruggere i pesi ImageNet nelle prime iterazioni quando i gradienti sono forti e non calibrati.

Dall'epoca 11 in poi il backbone è scongelato con LR ridotto di un fattore `BACKBONE_LR_FACTOR = 0.1` (fine-tuning). Questo schema è una buona pratica consolidata nel transfer learning.

### Scheduler

`ReduceLROnPlateau(mode='min', factor=0.5, patience=5)` sulla val loss. Se la val loss non migliora per 5 epoche consecutive, il LR viene dimezzato.

### Metriche tracciate (WandB)

| Metrica | Descrizione |
|---|---|
| `train/loss`, `val/loss` | Loss combinata (rot + λ·trans) |
| `train/deg`, `val/deg` | Errore angolare medio in gradi (quaternione) |
| `train/t_mse`, `val/t_mse` | MSE sulla traslazione (m²) |
| `backbone_unfrozen` | Flag 0/1 per monitorare il cambio di fase |

### Avvio

```bash
python trainBaseline.py
```

Il checkpoint viene salvato ad ogni epoca in `pose_resnet50_baseline_checkpoint.pth`. Il miglior modello (min val loss) viene salvato in `pose_resnet50_baseline_best.pth`.

---

## Metriche di Valutazione

### ADD — Average Distance of Model Points

Metrica standard per 6D pose estimation su LineMod.

Per ogni sample, si trasformano N punti 3D del modello CAD (campionati dal `.ply`) con la posa GT e la posa predetta, e si misura la distanza media tra i punti corrispondenti:

```
ADD = mean_i( || (R_gt · p_i + T_gt) − (R_pred · p_i + T_pred) || )
```

**Soglia:** un sample è "success" se `ADD < 0.1 · diameter` (10% del diametro dell'oggetto). Il diametro è letto da `models/models_info.yml`.

### ADD-S — per oggetti simmetrici

Eggbox (`obj_id=10`) e glue (`obj_id=11`) sono oggetti simmetrici: ruotati di 180° attorno a certi assi, sembrano identici. Con ADD standard, una posa corretta ma speculare viene penalizzata come se fosse completamente sbagliata.

ADD-S sostituisce la corrispondenza punto-per-punto con la **distanza al punto più vicino**:

```
ADD-S = mean_i( min_j( || p_pred_i − p_gt_j || ) )
```

**Implementazione in `evaluate_metricsBaseline.py`:**
- `SYMMETRIC_OBJ_IDS = {10, 11}` — oggetti che usano ADD-S
- `adds_distance(pts, R_gt, T_gt, R_pred, T_pred)` — calcola ADD-S
- `add_distance(pts, R_gt, T_gt, R_pred, T_pred)` — calcola ADD
- `pose_error(pts, R_gt, T_gt, R_pred, T_pred, obj_id)` — dispatcha ADD-S per `obj_id ∈ {10,11}`, ADD altrimenti

### Nota sulla complessità di ADD-S

Con N=500 punti, ADD-S crea una matrice di distanze pairwise `(500, 500)` per ogni sample. Il costo è accettabile per evaluation ma potrebbe rallentare il training se usato come loss — per questo in training si usa ADD standard su tutti gli oggetti (la loss è già descritta sopra, non usa ADD).

---

## Evaluation (`evaluate_metricsBaseline.py`)

Lo script valuta il modello su 4 modalità per isolare le diverse sorgenti di errore:

| Modalità | R da | T da | Scopo |
|---|---|---|---|
| **GT Crop** | GT bbox crop | T_gt | Isola la qualità di rotazione pura |
| **YOLO + T_gt** | YOLO bbox crop | T_gt | Misura l'impatto del crop YOLO su R |
| **YOLO + T_pinhole** | YOLO bbox crop | Geometrica (pinhole) | Sistema geometrico completo |
| **YOLO + T_pred** | YOLO bbox crop | Predetta dal modello | Sistema appreso completo |

**Traslazione pinhole:** dalla bbox YOLO si stima Z via `Z = fx * diameter / pixel_size`, poi `X = (u_center - cx) * Z / fx` e `Y = (v_center - cy) * Z / fy`. È una stima geometrica che dipende dalla conoscenza del diametro reale dell'oggetto — non è appresa.

**Traslazione predetta:** output del `tvec_head` convertito in mm (`* 1000.0`) per confronto coerente con le distanze ADD in mm.

```bash
python evaluate_metricsBaseline.py
```

Output: tabella con 4 colonne per ogni classe + media globale, con asterisco (*) per le classi con ADD-S.

---

## Note Tecniche

### Dominio Gap train/eval
In training si usa il GT bounding box per croppare l'oggetto. In evaluation end-to-end si usa il bbox predetto da YOLO. Questo introduce un dominio gap inevitabile: il YOLO può dare bbox imprecise o mancate. La modalità "YOLO + T_gt" quantifica esattamente questo gap sulla componente di rotazione.

### Conversione quaternione → matrice di rotazione
`scipy.spatial.transform.Rotation.from_quat(q).as_matrix()` — la convenzione di scipy per i quaternioni è `(x, y, z, w)`, coerente con `matrix_to_quaternion` in `utils/resNetUtils.py`.

### Traslazione GT
La traslazione GT in `gt.yml` (`cam_t_m2c`) è in **millimetri**. Il dataset la restituisce in mm; il training la converte in metri per la loss. L'evaluator usa mm per il calcolo ADD (coerente con i punti PLY, anch'essi in mm).
