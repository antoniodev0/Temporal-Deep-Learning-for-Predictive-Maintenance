# Deep Learning Temporale per la Manutenzione Predittiva
## LSTM vs Transformer per la Predizione della Vita Utile Residua (RUL) su CMAPSS

> Studio comparativo sistematico delle architetture LSTM e Transformer per la predizione della RUL sul dataset benchmark NASA CMAPSS.

---

## Indice

- [Panoramica del Progetto](#panoramica-del-progetto)
- [Dataset](#dataset)
- [Struttura della Repository](#struttura-della-repository)
- [Setup dell'Ambiente](#setup-dellambiente)
- [Descrizione dei File Sorgente](#descrizione-dei-file-sorgente)
- [Descrizione dei Notebook](#descrizione-dei-notebook)
- [Architetture dei Modelli](#architetture-dei-modelli)
- [Metriche di Valutazione](#metriche-di-valutazione)
- [Risultati](#risultati)
- [Analisi e Conclusioni](#analisi-e-conclusioni)
- [Riferimenti](#riferimenti)

---

## Panoramica del Progetto

Questo progetto confronta le architetture **LSTM** (Long Short-Term Memory) e **Transformer** per il task di regressione della **Vita Utile Residua (RUL)** in scenari di manutenzione predittiva industriale.

La RUL di un motore è il numero di cicli operativi rimanenti prima del guasto. Predirla con precisione permette di pianificare la manutenzione evitando sia guasti improvvisi (troppo tardi) sia interventi inutili (troppo presto).

Mentre i Transformer hanno rivoluzionato NLP e computer vision, il loro vantaggio rispetto alle reti ricorrenti su **serie temporali industriali brevi e strutturate** rimane una questione aperta. Questo progetto affronta il problema con esperimenti controllati sul benchmark CMAPSS.

---

## Dataset

**NASA CMAPSS** — Commercial Modular Aero-Propulsion System Simulation

Il dataset simula il degrado di motori turboventola fino al guasto. Contiene 4 sotto-dataset con diversi livelli di complessità:

| Sotto-dataset | Motori train | Motori test | Condizioni operative | Modi di guasto |
|---------------|-------------|-------------|----------------------|----------------|
| FD001         | 100         | 100         | 1                    | 1              |
| FD002         | 260         | 259         | 6                    | 1              |
| FD003         | 100         | 100         | 1                    | 2              |
| FD004         | 249         | 248         | 6                    | 2              |

Ogni file contiene 26 colonne: ID motore, ciclo operativo, 3 impostazioni operative e 21 sensori. Le ultime 2 colonne (sempre NaN) vengono scartate.

### Impostazioni operative (setting1, setting2, setting3)

Le tre impostazioni operative descrivono le **condizioni di volo** del motore in ogni ciclo:

| Colonna | Grandezza fisica |
|---------|-----------------|
| setting1 | Quota di volo (ft) |
| setting2 | Numero di Mach |
| setting3 | Angolo del throttle (TRA, %) |

In **FD001 e FD003** (1 condizione operativa) questi valori sono quasi costanti ad ogni ciclo — il motore opera sempre nello stesso regime di volo. In **FD002 e FD004** (6 condizioni operative) variano tra i cicli perché la simulazione alterna 6 profili di volo diversi. Questa variabilità è la ragione per cui in FD002/FD004 è necessaria la normalizzazione cluster-based: lo stesso sensore ha range di valori molto diversi a seconda della quota e della velocità.

### Strategia di etichettatura RUL

Si adotta la **RUL lineare a tratti** (piecewise linear / clipped): la RUL massima viene cappata a 125 cicli, assumendo che il motore sia in stato sano nella prima parte della sua vita. Questo è lo standard della letteratura e migliora significativamente le prestazioni.

### Selezione delle feature

La selezione è stata fatta in due fasi:

**1. Analisi della varianza (notebook `01_eda.ipynb`)**
Nel notebook EDA viene calcolata la varianza di ogni sensore su FD001 e visualizzata in ordine crescente. I sensori con varianza **esattamente zero o trascurabile** sono quelli che non cambiano mai — né tra cicli, né tra motori diversi. La selezione finale segue lo standard della letteratura (Zheng et al., 2017), che identifica 7 sensori da rimuovere:

```
s1=0.0, s5=0.0, s6=0.0, s10=0.0, s16=0.0, s18=0.0, s19=0.0  → rimossi
```

**Nota importante**: altri sensori come `s8`, `s13`, `s15` hanno varianza molto bassa ma **non vengono rimossi** perché mostrano comunque un trend correlato al degrado (varianza piccola ≠ segnale inutile). La soglia non è un criterio automatico: la selezione finale rispecchia la scelta consolidata in letteratura, confermata visivamente nell'analisi dei trend.

**2. Rimozione automatica al caricamento (`src/preprocessing.py`, funzione `load_raw`)**
Una volta identificati nell'EDA, i sensori da rimuovere vengono definiti come costante `DROP_SENSORS` e rimossi automaticamente ad ogni caricamento del dataset, **prima di qualsiasi altra elaborazione**:

```python
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

def load_raw(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df.drop(columns=DROP_SENSORS, inplace=True)  # ← qui avviene la rimozione
    return df
```

**Feature finali**: 14 sensori (`s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`) + 3 impostazioni operative = **17 feature totali** → dimensione dell'input `(window_size, 17)`.

Per FD002 e FD004 (più condizioni operative) si applica la **normalizzazione cluster-based**: le condizioni operative vengono clusterizzate con KMeans (k=6) e la normalizzazione MinMax viene applicata separatamente per ogni cluster.

---

## Struttura della Repository

```
Temporal-Deep-Learning-for-Predictive-Maintenance/
│
├── data/
│   ├── raw/                        # File .txt originali CMAPSS
│   └── processed/                  # Array numpy preprocessati (.npy)
│
├── notebooks/
│   ├── 01_eda.ipynb                # Analisi esplorativa dei dati
│   ├── 02_preprocessing.ipynb      # Pipeline di preprocessing
│   ├── 03_lstm.ipynb               # Training e valutazione LSTM
│   ├── 04_transformer.ipynb        # Training e valutazione Transformer
│   └── 05_comparison.ipynb         # Confronto finale, plot e analisi
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Logica di preprocessing (windowing, normalizzazione)
│   ├── metrics.py                  # Implementazione RMSE, MAE, NASA Score
│   ├── training.py                 # Loop di training, callback, runner esperimenti
│   └── models/
│       ├── __init__.py
│       ├── lstm_model.py           # Architettura LSTM
│       └── transformer_model.py    # Architettura Transformer con PE e MHA
│
├── experiments/
│   ├── configs/
│   │   ├── lstm_config.yaml        # Iperparametri LSTM
│   │   └── transformer_config.yaml # Iperparametri Transformer
│   └── results/
│       ├── main_comparison.csv     # Metriche principali (RMSE, MAE, NASA Score)
│       └── rul_range_analysis.csv  # Analisi MAE per fascia di RUL
│
├── saved_models/                   # Checkpoint Keras (.keras) dei modelli migliori
├── plots/                          # Figure di output (curve, scatter, distribuzioni)
├── venv/                           # Ambiente virtuale Python 3.11
├── requirements.txt
└── README.md
```

---

## Setup dell'Ambiente

### Requisiti

- Python 3.11 (via `brew install python@3.11`)
- macOS con Apple Silicon (M1/M2/M3) — usa Metal GPU via `tensorflow-metal`

### Installazione

```bash
# Creare l'ambiente virtuale con Python 3.11
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv venv
source venv/bin/activate

# Installare le dipendenze
pip install -r requirements.txt

# Registrare il kernel Jupyter
python -m ipykernel install --user --name rul-pred --display-name "Python (rul-pred)"
```

### Avviare Jupyter

```bash
source venv/bin/activate
jupyter notebook
```

Selezionare il kernel **"Python (rul-pred)"** in ogni notebook.

---

## Descrizione dei File Sorgente

### `src/preprocessing.py`

Contiene l'intera pipeline di preprocessing:

- `load_raw(path)` — carica un file .txt CMAPSS, assegna i nomi alle colonne e rimuove i sensori costanti
- `add_rul(df)` — calcola e aggiunge la colonna RUL con cap a 125 cicli
- `normalize_standard(train_df, test_df, feature_cols)` — normalizzazione MinMax (fit sul train, transform su entrambi); per FD001 e FD003
- `normalize_clustered(train_df, test_df, feature_cols)` — normalizzazione MinMax per cluster KMeans (k=6); per FD002 e FD004
- `make_windows(df, feature_cols, window_size)` — genera finestre scorrevoli; output `X` di forma `(n_campioni, window_size, n_feature)` e `y` scalare (RUL all'ultimo timestep)
- `make_test_windows(df, feature_cols, window_size)` — estrae l'ultima finestra per ogni motore nel test set
- `get_feature_cols(df)` — restituisce le colonne feature (esclude engine_id, cycle, rul)

### `src/metrics.py`

Implementa le tre metriche di valutazione:

- `rmse(y_true, y_pred)` — Root Mean Squared Error
- `mae(y_true, y_pred)` — Mean Absolute Error
- `nasa_score(y_true, y_pred)` — Funzione di scoring asimmetrica NASA: penalizza maggiormente le predizioni tardive (errore > 0) rispetto a quelle precoci
- `evaluate(y_true, y_pred)` — restituisce un dizionario con tutte e tre le metriche
- `evaluate_by_rul_range(y_true, y_pred)` — calcola MAE e RMSE separatamente per le fasce 0–50, 50–100, 100–125

### `src/models/lstm_model.py`

Definisce `build_lstm(window_size, n_features, units, dropout, learning_rate)`.
Architettura: `Input → LSTM(units, return_sequences=True) → Dropout → LSTM(units//2) → Dropout → Dense(64, relu) → Dense(1, linear)`.

### `src/models/transformer_model.py`

Definisce tre componenti:

- `PositionalEncoding` — codifica posizionale sinusoidale fissa (nessun parametro apprendibile), decorata con `@keras.saving.register_keras_serializable` per la corretta serializzazione
- `TransformerEncoderBlock` — blocco encoder standard: MultiHeadAttention → Add & Norm → FFN → Add & Norm
- `build_transformer(window_size, n_features, d_model, num_heads, ff_dim, num_blocks, dropout, learning_rate)` — assembla il modello completo con proiezione lineare iniziale, PE, N blocchi encoder, GlobalAveragePooling1D e testa regressiva

### `src/training.py`

- `get_callbacks(model_path, patience)` — restituisce `[EarlyStopping, ModelCheckpoint, ReduceLROnPlateau]`
- `train_model(...)` — esegue il training con 80/20 split di validazione
- `run_experiment(...)` — esegue training + inference + calcolo metriche in un unico passo; carica il modello migliore salvato dal checkpoint per la valutazione finale

---

## Descrizione dei Notebook

### `01_eda.ipynb` — Analisi Esplorativa

Carica tutti e 4 i sotto-dataset e visualizza:
- Distribuzione della lunghezza del ciclo di vita dei motori per sotto-dataset
- Varianza dei sensori per identificare quelli costanti da rimuovere
- Andamento dei sensori nel tempo per un singolo motore (FD001)

### `02_preprocessing.ipynb` — Pipeline di Preprocessing

Esegue la pipeline completa su tutti e 4 i sotto-dataset:
1. Caricamento e aggiunta RUL
2. Normalizzazione (standard per FD001/FD003, cluster-based per FD002/FD004)
3. Generazione finestre scorrevoli (`window_size=30`)
4. Salvataggio degli array in `data/processed/` come file `.npy`

Output: 4 array per dataset (`X_train`, `y_train`, `X_test`, `y_test`), verificati per forma.

### `03_lstm.ipynb` — Modello LSTM

Per ciascuno dei 4 sotto-dataset:
- Carica gli array preprocessati
- Costruisce e addestra il modello LSTM (128 unità, dropout 0.2, lr 1e-3)
- Salva il miglior modello in `saved_models/lstm_{FD}.keras`
- Stampa le metriche sul test set
- Produce le curve di training/validation loss

### `04_transformer.ipynb` — Modello Transformer

Stesso flusso del notebook LSTM, con il Transformer (d_model=64, 4 head, ff_dim=128, 2 blocchi encoder). Salva i modelli in `saved_models/transformer_{FD}.keras`.

### `05_comparison.ipynb` — Confronto Finale

- Carica tutti gli 8 modelli salvati (2 architetture × 4 dataset)
- Calcola RMSE, MAE e NASA Score per tutti
- Salva i risultati in `experiments/results/main_comparison.csv`
- Produce scatter plot predetto vs reale per tutte le combinazioni
- Esegue l'analisi per fascia di RUL (Esperimento 3) e salva in `rul_range_analysis.csv`

---

## Grafici e Analisi Visiva

### `plots/01_lifecycle_distribution.png` — Distribuzione del ciclo di vita dei motori

![Lifecycle distribution](plots/01_lifecycle_distribution.png)

Il grafico mostra, per ciascuno dei 4 sotto-dataset, l'istogramma del numero massimo di cicli operativi raggiunti da ogni motore prima del guasto.

**Osservazioni:**
- **FD001 e FD002** presentano distribuzioni più concentrate e simmetriche, con un picco attorno a 175–200 cicli e una coda destra che arriva fino a ~350–375 cicli.
- **FD003 e FD004** hanno distribuzioni molto più disperse, con motori che arrivano fino a 500+ cicli. La coda destra è marcata, segno che alcuni motori durano quasi il doppio della media. Questo rende la predizione della RUL più difficile: la variabilità inter-motore è molto alta.
- Il cap a 125 cicli applicato alla RUL è giustificato dal fatto che la maggior parte dei motori ha una fase "sana" iniziale ben superiore ai 125 cicli, durante la quale il degrado non è ancora rilevabile dai sensori.

---

### `plots/01_sensor_trends.png` — Andamento dei sensori nel tempo (Motore 1, FD001)

![Sensor trends](plots/01_sensor_trends.png)

Il grafico mostra l'evoluzione nel tempo dei 14 sensori mantenuti e delle 3 impostazioni operative per un singolo motore di FD001 (~200 cicli di vita).

**Osservazioni:**
- **setting1 e setting2** appaiono come segnali rumorosi senza trend visibile: rappresentano le condizioni di volo che variano ad ogni ciclo indipendentemente dallo stato del motore. In FD001 (1 sola condizione operativa) la varianza è molto ridotta.
- **setting3** è pressoché costante a ~100: non porta informazione sul degrado ma viene mantenuto perché utile per la normalizzazione cluster-based di FD002/FD004.
- **s2, s3, s4** mostrano un chiaro trend crescente verso fine vita: sono i sensori più informativi per il degrado.
- **s11** mostra un pattern complesso: decresce nella fase iniziale per poi stabilizzarsi, con un cambio di regime nel mezzo della vita utile.
- **s12 e s13** mostrano trend opposti (uno crescente, l'altro decrescente), coerentemente con la fisica del motore (pressione/temperatura che aumentano mentre il rendimento diminuisce).
- **s7, s8, s15** sono prevalentemente rumorosi con trend deboli: contribuiscono marginalmente ma vengono mantenuti perché la loro rimozione peggiora le prestazioni in letteratura.

Questo grafico motiva la scelta di una finestra temporale (window_size=30): un singolo timestep non contiene abbastanza informazione sul trend, ma una finestra di 30 cicli cattura già i pattern di degrado più significativi.

---

### `plots/03_lstm_training_curves.png` — Curve di training LSTM

![LSTM training curves](plots/03_lstm_training_curves.png)

Le curve mostrano l'evoluzione della MSE Loss su train (blu) e validation (arancione) per i 4 dataset.

**Osservazioni:**
- Tutte e 4 le curve presentano una caratteristica struttura **a gradino**: un primo plateau attorno a epoch 5–10, seguito da una discesa brusca (intorno a epoch 25–30 per FD001/FD003, prima per FD002/FD004). Questo è tipico delle LSTM: la rete prima apprende le statistiche di base del segnale, poi "scopre" la struttura temporale del degrado.
- **Train e validation si sovrappongono quasi perfettamente** per tutti i dataset: nessun overfitting. L'EarlyStopping con patience=15 ha interrotto il training nel momento giusto.
- **FD001 e FD003** (mono-condizione) richiedono più epoche (~85–90) rispetto a **FD002 e FD004** (~45–75). Paradossalmente i dataset più complessi convergono prima in termini di epoche, perché il maggior numero di campioni (46k e 54k vs 17k e 21k) velocizza l'aggiornamento dei pesi.
- I valori finali di val_loss sono coerenti con i RMSE riportati nella tabella dei risultati (val_loss ≈ RMSE²).

---

### `plots/04_transformer_training_curves.png` — Curve di training Transformer

![Transformer training curves](plots/04_transformer_training_curves.png)

**Osservazioni:**
- Il Transformer converge **molto più rapidamente** dell'LSTM: tutti i dataset raggiungono la zona di plateau entro le prime 5–10 epoche, senza il gradino a due stadi caratteristico dell'LSTM. Il meccanismo di attention permette di catturare immediatamente le dipendenze a lungo raggio nella finestra.
- **FD001, FD002, FD003**: le curve train/val sono allineate e lisce, con EarlyStopping che interviene tra epoch 36 e 52.
- **FD004**: si osserva una leggera oscillazione nella fase finale (epoch 30–40) con un piccolo divario tra train e val. Questo è il segnale di una lieve instabilità su questo dataset, il più complesso (6 condizioni + 2 modi di guasto). Il Transformer fatica a generalizzare su distribuzioni non stazionarie, come confermato dall'RMSE finale di 39.4 contro i 28.7 dell'LSTM.
- Il Transformer converge in **meno epoche totali** ma con **loss finale più alta su FD004**, evidenziando un trade-off tra velocità di convergenza e qualità della soluzione trovata.

---

### `plots/05_scatter_pred_vs_true.png` — Predetto vs Reale (scatter plot)

![Scatter predicted vs true](plots/05_scatter_pred_vs_true.png)

Ogni punto rappresenta un motore del test set. L'asse X è la RUL reale (da `RUL_FDxxx.txt`), l'asse Y è la RUL predetta dal modello. La linea rossa tratteggiata è la bisettrice perfetta (predizione = realtà).

**Osservazioni:**
- **LSTM su FD001 e FD003**: i punti sono ben allineati alla diagonale, con dispersione ridotta. Il modello è sia accurato sia calibrato: non tende sistematicamente a sovra- o sottostimare.
- **LSTM su FD002 e FD004**: maggiore dispersione attorno alla diagonale, coerente con l'RMSE più alto. La distribuzione degli errori appare comunque bilanciata: errori precoci e tardivi sono presenti in misura simile.
- **Transformer su FD001, FD002, FD003**: scatter simile all'LSTM, con punti distribuiti attorno alla diagonale senza bias evidenti.
- **Transformer su FD004**: il grafico è chiaramente peggiore degli altri. Si notano diversi punti con RUL reale bassa (0–50 cicli) ma RUL predetta alta (80–120 cicli) — esattamente le predizioni tardive che il NASA Score penalizza in modo esponenziale. Questo spiega il NASA Score di 2.369.435 (contro 7.771 dell'LSTM): un numero relativamente piccolo di predizioni gravemente tardive domina il risultato.

---

## Architetture dei Modelli

### LSTM

```
Input (30, 17)
    │
LSTM(128, return_sequences=True)
    │
Dropout(0.2)
    │
LSTM(64)
    │
Dropout(0.2)
    │
Dense(64, relu)
    │
Dense(1, linear)  ← predizione RUL
```

### Transformer

```
Input (30, 17)
    │
Dense(64)  ← proiezione lineare
    │
PositionalEncoding (sinusoidale, fissa)
    │
TransformerEncoderBlock × 2
  [MultiHeadAttention(4 head) → Add & Norm → FFN(128) → Add & Norm]
    │
GlobalAveragePooling1D
    │
Dense(64, relu)
    │
Dropout(0.1)
    │
Dense(1, linear)  ← predizione RUL
```

---

## Metriche di Valutazione

### RMSE — Root Mean Squared Error

Penalizza gli errori grandi in modo quadratico. Metrica principale per il confronto.

### MAE — Mean Absolute Error

Errore medio assoluto. Più interpretabile del RMSE, meno sensibile agli outlier.

### NASA Score (funzione asimmetrica)

```
d = y_pred - y_true
score = sum(exp(-d/13) - 1  se d < 0   ← predizione anticipata
            exp(d/10) - 1   se d ≥ 0)  ← predizione tardiva
```

Penalizza le **predizioni tardive** (sovrastimare la RUL rimasta) più severamente di quelle precoci. Riflette il costo reale asimmetrico: predire un guasto troppo tardi causa danni al motore, predirlo troppo presto causa manutenzione inutile. Punteggio più basso = migliore.

---

## Risultati

### Confronto Principale (Esperimento 1)

| Modello     | FD001 RMSE | FD002 RMSE | FD003 RMSE | FD004 RMSE |
|-------------|-----------|-----------|-----------|-----------|
| LSTM        | **14.82** | **29.29** | 15.61     | **28.70** |
| Transformer | 15.50     | 29.45     | **15.26** | 39.36     |

| Modello     | FD001 MAE | FD002 MAE | FD003 MAE | FD004 MAE |
|-------------|-----------|-----------|-----------|-----------|
| LSTM        | 10.92     | 19.03     | 11.71     | **20.41** |
| Transformer | 11.75     | **18.44** | **11.47** | 23.97     |

| Modello     | FD001 Score | FD002 Score | FD003 Score | FD004 Score |
|-------------|------------|------------|------------|------------|
| LSTM        | **387**    | **59.884** | **393**    | **7.771**  |
| Transformer | 489        | 367.973    | 407        | 2.369.435  |

*Valori in grassetto = migliore per colonna. NASA Score: più basso è meglio.*

### Analisi per Fascia di RUL — MAE (Esperimento 3)

| Fascia RUL | LSTM MAE (FD001) | Transformer MAE (FD001) |
|------------|-----------------|------------------------|
| 0–50       | **3.93**        | 4.42                   |
| 50–100     | **16.43**       | 18.00                  |
| 100–125    | **7.25**        | 9.37                   |

---

## Analisi e Conclusioni

### LSTM vince su FD002 e FD004 (multi-condizione)

Nei sotto-dataset con 6 condizioni operative, LSTM mantiene prestazioni solide mentre il Transformer degrada significativamente (FD004 RMSE: 28.7 vs 39.4). Questo suggerisce che la normalizzazione cluster-based mitiga, ma non elimina, la difficoltà del Transformer nel gestire distribuzioni non stazionarie senza un meccanismo di memoria esplicito.

### Transformer competitivo su FD001 e FD003 (mono-condizione)

Su dataset più semplici e stazionari, le due architetture sono sostanzialmente equivalenti in RMSE e MAE. Il Transformer mostra un leggero vantaggio in MAE su FD003, ma LSTM rimane superiore nel NASA Score per tutti i dataset.

### NASA Score: LSTM sistematicamente migliore

LSTM produce sistematicamente un NASA Score più basso (migliore) su tutti e 4 i dataset. Questo indica che LSTM tende a essere più conservativo: quando sbaglia, sbaglia in anticipo (stima RUL minore del reale) piuttosto che in ritardo. Il Transformer invece accumula errori tardivi severi, in particolare su FD002 e FD004.

### Comportamento per fascia di RUL

Entrambi i modelli commettono gli errori maggiori nella fascia 50–100 cicli. Questa è la fase di transizione dalla zona "sana" alla zona di degrado accelerato, dove il segnale dei sensori è più ambiguo. LSTM performa meglio di Transformer in tutte le fasce.

### Considerazioni finali

Per la manutenzione predittiva su dati CMAPSS, **LSTM è preferibile** sia per le prestazioni medie superiori sia per il profilo di rischio più favorevole (NASA Score). Il Transformer potrebbe beneficiare di tecniche di adattamento al dominio o di architetture ibride (es. patch tokenization come in PatchTST) per sfruttare meglio l'attention su serie temporali industriali brevi.

---

## Riferimenti

1. Saxena, A., et al. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation.* IMAS.
2. Zheng, S., et al. (2017). *Long Short-Term Memory Network for Remaining Useful Life Estimation.* IMAS.
3. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
4. Wu, N., et al. (2020). *Deep Transformer Models for Time Series Forecasting.* arXiv:2001.08317.
5. Li, X., et al. (2018). *Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks.* Reliability Engineering & System Safety.

---

*Progetto sviluppato per il corso di Deep Learning — Laurea Magistrale in Ingegneria Informatica.*
