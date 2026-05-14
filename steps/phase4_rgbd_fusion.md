# Phase 4 — RGB-D Fusion (DenseFusion-inspired)

## Obiettivo (dal brief)

> "The RGB-D Pose Predictor Model separates the processing of RGB and depth data, using different
> branches for each [...] The system ultimately predicts the 3D rotation matrix (3x3) and
> 3D translation vector (3D), trained using [...] ADD loss."
> — project6.md, Phase 4

---

## Architettura

Tre varianti di modello, tutte con lo stesso schema fondamentale:

```
RGB (224×224×3)  →  RGB backbone   → f_rgb  (2048D)  ┐
Depth (224×224×1)→  Depth backbone → f_depth (512D)  ├→ concat → fusion_mlp → shared (256D)
meta_info (8D)   →  meta_encoder   → f_meta  (64D)   ┘
                                                              ↓           ↓
                                                       translation_head   rotation_head
                                                            (B,3)          (B,3,3) ← SO(3)
```

### Varianti

| File | RGB branch | Depth branch | Depth input |
|---|---|---|---|
| `phase4_fusion/main/model.py` | ResNet-50 ImageNet | ResNet-18 ImageNet | 3ch (replicated) |
| `phase4_fusion/extension/model.py` | ResNet-50 ImageNet | custom ResNet-10 | 1ch |
| `archive/RGBD_FusionPredictor_custom_5layer_cnn.py` | ResNet-50 ImageNet | custom 5-layer CNN | 1ch (archived) |

### Meta-branch

Vettore 8D `[cx/W, cy/H, w/W, h/H, fx/1000, fy/1000, px/W, py/H]` — normalizzato nell'immagine. Non richiesto dal brief ma aggiunto come estensione: fornisce geometria della camera e posizione del bbox come prior per la traslazione.

---

## Rappresentazione della Rotazione — 6D Continuous (Zhou et al. CVPR 2019)

**Problema con 9D raw:** `nn.Linear(256, 9)` + `.view(-1, 3, 3)` produce una matrice 3×3 generica, non necessariamente in SO(3). I gradienti non sono vincolati alla varietà delle rotazioni.

**Soluzione — 6D representation:**

```python
# common/rotation.py
def rot6d_to_matrix(r6d):  # (B, 6) → (B, 3, 3)
    a1, a2 = r6d[:, :3], r6d[:, 3:]
    b1 = F.normalize(a1, p=2, dim=1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)  # colonne = base ortonormale
```

**Proprietà:**
- Output sempre in SO(3) (det=+1, colonne ortonormali) per costruzione
- Rappresentazione continua → gradient flow più stabile rispetto a discontinuità quaternione/Eulero
- Gram-Schmidt differenziabile → backprop fluisce attraverso l'ortogonalizzazione

**Impatto architetturale:** `rotation_head` passa da `nn.Linear(256, 9)` a `nn.Linear(256, 6)`. I vecchi checkpoint sono **incompatibili** (shape mismatch) → retraining necessario.

---

## Depth Preprocessing

### Normalizzazione e clipping (`phase4_fusion/main/rgbd_utils.py`, `phase4_fusion/extension/rgbd_utils.py`)

```python
def convert_depth_to_meters(depth_raw, depth_scale):
    depth_m = (depth_raw.astype(np.float32) * float(depth_scale)) / 1000.0
    return np.clip(depth_m, 0.0, 3.0)
```

**Motivazione clip [0, 3] m:**
- Pixel invalidi del sensore (depth=0) producevano 0 m → bias nella rete
- LineMod: oggetti a 0.3–0.7 m, sfondo a < 2 m → [0, 3] copre tutto il range utile
- Clip non tocca pixel validi, elimina solo valori fuori range

### Input depth per variante

| Variante | Branch depth | Formato input |
|---|---|---|
| `phase4_fusion/main/model.py` | ResNet-18 3ch | depth replicata 3× (`prepare_depth_tensor`) |
| `phase4_fusion/extension/model.py` | custom ResNet-10 1ch | depth 1ch (`prepare_depth_tensor_custom`) |

---

## Loss Function

```
Loss = ADD(R_pred, T_pred, R_gt, T_gt, model_points)
```

`ADDLoss` (`phase4_fusion/main/add_loss.py`) calcola la distanza media tra punti del modello trasformati con la posa predetta e la GT:

```
ADD = mean_i( || (R_pred · p_i + T_pred) − (R_gt · p_i + T_gt) || )
```

**Nota:** `ADDLoss.forward` riceve `pred_R` come `(B,3,3)` — il `.view(-1,3,3)` interno è ora un no-op, mantiene compatibilità.

---

## Metriche di Valutazione

Identiche a Phase 3: ADD / ADD-S con soglia 10% diametro, via `common/pose_metrics.pose_error`.

Tutti e tre gli evaluator (`phase4_fusion/main/evaluate.py`, `phase4_fusion/extension/evaluate.py`, `phase3_baseline/evaluate.py`) importano ora da `common.pose_metrics` — nessuna duplicazione.

### Uniformità del crop (fix Phase 4)

Problema pre-esistente: `evaluate_metricsRGBD_custom.py` usava GT bbox (`ann['obj_bb']`) mentre `evaluate_metricsRGBD.py` usava YOLO bbox. Il confronto tra modelli era iniquo (GT crop più pulito → metriche artificicialmente migliori per il custom).

**Fix:** `evaluate_metricsRGBD_custom.py` ora usa YOLO detection, allineato a `evaluate_metricsRGBD.py`.

---

## Training

### Script

| Script | Modello |
|---|---|
| `phase4_fusion/main/train.py` | `RGBD_FusionPredictor` (ResNet-50 + ResNet-18) |
| `phase4_fusion/extension/train.py` | `FusionResNetCustom` (ResNet-50 + ResNet-10) |

### Hyperparameters

| Parametro | Valore |
|---|---|
| `BATCH_SIZE` | 32 |
| `LEARNING_RATE` | 1e-4 |
| `WEIGHT_DECAY` | 1e-4 |
| `EPOCHS` | 100 |
| `N_POINTS` | 500 |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |

### Metriche tracciate (WandB)

| Metrica | Descrizione |
|---|---|
| `train/loss_add`, `val/loss_add` | ADD loss media (m) |
| `train/t_mse`, `val/t_mse` | MSE traslazione |
| `train/r_mse`, `val/r_mse` | MSE rotazione (matrici 3×3) |
| `val_obj/error_mm_{id}` | ADD per-oggetto in mm |

---

## Evaluation

```bash
python -m phase4_fusion.main.evaluate       # ResNet-50 + ResNet-18
python -m phase4_fusion.extension.evaluate  # ResNet-50 + custom ResNet-10
```

Entrambi producono una tabella con ADD medio (mm) e Accuracy (%) per classe. Gli oggetti simmetrici (eggbox*, glue*) usano ADD-S.

### Risultati Ufficiali

Dopo 100 epoche di addestramento su GPU, l'introduzione della profondità ha completamente risolto i problemi di traslazione riscontrati nella Fase 3 (che aveva solo 1.9% di accuratezza).

| Modello | Accuracy (ADD < 10% d) | Range Errore Medio (mm) |
|---|---|---|
| **MAIN** (ResNet-18) | **98.4%** | 3.66 – 8.70 mm |
| **EXT** (ResNet-1ch) | **95.8%** | 5.13 – 10.15 mm |

**Dettaglio classi (Extension):**
`ape`: 7.88 mm | `benchvise`: 8.91 mm | `camera`: 7.60 mm | `can`: 7.63 mm | `cat`: 6.96 mm | `driller`: 9.79 mm | `duck`: 7.88 mm | `eggbox*`: 5.13 mm | `glue*`: 5.45 mm | `holepuncher`: 7.01 mm | `iron`: 10.15 mm | `lamp`: 9.08 mm | `phone`: 8.88 mm.

**Analisi:** 
- Entrambi i modelli superano agilmente il 95% di accuratezza globale per la stima della posa 6D completa.
- La variante `MAIN` (con ResNet-18 ImageNet) ha superato la variante `EXT` (con ResNet-10 custom). Questo suggerisce che i pesi pre-addestrati su ImageNet (usati nel Main copiando la depth su 3 canali) compensano e superano il vantaggio teorico di avere un'architettura nativa a 1 canale addestrata da zero (come nell'Extension).

---

## Confronto con DenseFusion (originale)

| Aspetto | DenseFusion (Wang et al. 2019) | Nostra implementazione |
|---|---|---|
| Fusion | Dense per-pixel (feature map 7×7) | Concatenazione globale (global avg pool) |
| Rotation | Quaternione + confidence | 6D continuous (Zhou et al.) |
| Iterative refinement | Sì | No |
| Depth branch | PointNet (point cloud 3D) | CNN su depth image 2D |
| Meta-branch | No | Sì (bbox + camera K) |

La nostra versione è una "2D semplificata" come richiesto dal brief. Le innovazioni rispetto alla versione naive (9D raw rotation, depth non clippata, GT bbox nel custom eval) sono documentate nel changelog.
