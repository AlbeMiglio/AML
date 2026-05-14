# Studio Approfondito — Fase 4: RGB-D Fusion

Questo documento è il tuo "manuale di studio" per la Fase 4. Proprio come per le Fasi 2 e 3, qui analizziamo il *perché* le cose sono state scritte in un certo modo nel codice, per prepararti al meglio per la stesura del report finale e per l'orale.

---

## 1. Il Problema Iniziale: Perché la Fase 4?
Nella Fase 3 (Baseline) abbiamo scoperto che una rete neurale che guarda solo un'immagine a colori (RGB) è bravissima a capire come è ruotato un oggetto, ma fallisce totalmente nel capire **a che distanza si trova** (la Traslazione).
Per risolvere questo problema, dobbiamo fornire alla rete il senso della profondità. Introduciamo quindi la mappa **Depth (D)**, trasformando il nostro approccio in **RGB-D**.

---

## 2. Le Due Varianti del Modello
Se guardi la cartella `phase4_fusion`, noterai che ci sono due sottocartelle: `main` ed `extension`. Questo perché abbiamo sviluppato due soluzioni diverse per trattare i dati di profondità.

### Variante A: `main` (Approccio Standard)
Usa una **ResNet-18** pre-addestrata per analizzare la profondità.
- *Problema:* La ResNet-18 si aspetta foto a colori (3 canali: R, G, B). La mappa di profondità invece è un solo canale (bianco/nero che indica la distanza).
- *Soluzione:* Nel codice (`dataset.py`) prendiamo la mappa di profondità e la "duplichiamo" su 3 canali identici per ingannare la rete e farle accettare l'input. È una tecnica veloce ed efficace.

### Variante B: `extension` (L'Approccio Custom/Avanzato)
Invece di "ingannare" una rete esistente, qui abbiamo scritto da zero una rete neurale su misura: la **`ResNet1ch`**.
- Questa rete è progettata per accettare nativamente immagini a **1 singolo canale**. È matematicamente e architetturalmente più elegante e risponde alla richiesta del progetto di proporre una "estensione" o "miglioramento".

---

## 3. L'Architettura: Come avviene la "Fusione"
Che tu scelga la variante Main o Extension, il concetto di fusione (`model.py`) è identico:

1. **Il ramo RGB:** Una ResNet-50 estrae 2048 numeri (feature) dai colori.
2. **Il ramo Depth:** La ResNet-18 (o la custom) estrae 512 numeri dalla geometria.
3. **Il ramo Meta:** Prende 8 numeri geometrici (come la posizione della Bounding Box e i parametri della fotocamera) e li trasforma in 64 numeri.
4. **La Fusione (Concatenazione):** Incolliamo tutti questi numeri insieme in un unico mega-vettore da 2624 numeri (`fused = torch.cat((f_rgb, f_depth, f_meta), dim=1)`).
5. **Le Teste Finali:** Questo mega-vettore viene passato a un gruppo di layer (MLP) che si divide nelle solite due teste: Traslazione e Rotazione.

> [!NOTE]
> Questa architettura è una versione "2D" e più leggera del famoso paper *DenseFusion*. Invece di fondere i dati pixel per pixel nello spazio 3D (che richiederebbe una potenza di calcolo assurda), noi estraiamo le feature globali in 2D e le fondiamo alla fine.

---

## 4. La grande novità: L'abbandono dei Quaternioni
Se apri `model.py`, noterai che la testa della rotazione non sputa più fuori 4 numeri (quaternione) come faceva nella Fase 3, ma ne sputa fuori **6**.

**Perché?**
Usare i quaternioni nelle reti neurali crea spesso problemi matematici (i gradienti si comportano male). Nel 2019, i ricercatori Zhou et al. hanno scoperto che rappresentare le rotazioni con **6 numeri (Continuous 6D Representation)** rende l'addestramento molto più stabile e preciso.
Il codice in `common/rotation.py` si occupa proprio di prendere questi 6 numeri e convertirli in una vera e propria matrice di rotazione 3x3 matematicamente perfetta (usando il processo di ortogonalizzazione di Gram-Schmidt).

---

## 5. Il preprocessing della Profondità (`rgbd_utils.py`)
I dati di profondità (Depth) del dataset LineMod sono grezzi. Non possiamo passarli alla rete così come sono.

```python
def convert_depth_to_meters(depth_raw, depth_scale):
    depth_m = (depth_raw.astype(np.float32) * float(depth_scale)) / 1000.0
    return np.clip(depth_m, 0.0, 3.0)
```
Cosa fa questo codice:
1. Converte i valori raw in veri **metri**.
2. **Effettua un clipping tra 0 e 3 metri**. Perché? Gli oggetti che ci interessano sono tutti sul tavolo, a circa mezzo metro dalla telecamera. Se c'è un muro sullo sfondo a 10 metri, quel numero "10" sballerebbe i calcoli della rete. Tagliando tutto a 3 metri, costringiamo la rete a ignorare lo sfondo lontano e a concentrarsi solo sugli oggetti vicini.

---

## 6. La Loss Function (`add_loss.py`)
A differenza della Fase 3 in cui calcolavamo due errori separati (uno per la rotazione e uno per la traslazione), qui usiamo direttamente la metrica ufficiale **ADD**.

```python
Loss = ADD(R_pred, T_pred, R_gt, T_gt, model_points)
```
In pratica:
1. Prendiamo i punti 3D del modello CAD dell'oggetto.
2. Li spostiamo nello spazio usando la posa *predetta* dalla nostra rete.
3. Li spostiamo nello spazio usando la posa *reale* (Ground Truth).
4. Calcoliamo la distanza tra questi due gruppi di punti. Quella distanza in millimetri è esattamente la nostra "Loss". Minimizzando questa loss, la rete impara a sovrapporre l'oggetto predetto a quello reale.

---

## Conclusioni per il Report
Quando scriverai il report finale, questi sono i punti di forza del vostro progetto da evidenziare per la Fase 4:
- Avete implementato **due varianti** architetturali (Standard ed Extension).
- Avete usato la rappresentazione **6D Continua** invece dei quaternioni classici, che è uno stato dell'arte moderno.
- Avete fuso informazioni **RGB**, **Depth** e persino metadati della **Fotocamera/BBox** per aiutare la rete a calcolare la traslazione in modo robusto.

### Analisi Empirica dei Risultati
I test finali (che hanno raggiunto il **98.4%** nel Main e il **95.8%** nell'Extension) dimostrano che la fusione RGB-D funziona e risolve completamente l'incapacità della Baseline di prevedere la traslazione. 
Un dato molto interessante (ottimo da discutere nel report) è che la variante **MAIN** (ResNet-18 standard ingannata con la Depth duplicata su 3 canali) ha battuto la variante **EXT** (ResNet-1ch custom). Questo accade spesso nel Deep Learning: i pesi pre-addestrati su milioni di immagini (ImageNet) racchiudono una conoscenza così vasta sui bordi e sulle forme, che riescono a superare il potenziale di una rete custom matematicamente più elegante ma addestrata da zero solo su poche migliaia di immagini.
