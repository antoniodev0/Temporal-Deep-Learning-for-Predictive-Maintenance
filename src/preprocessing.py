import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Nomi delle 26 colonne del formato CMAPSS (nessun header nel file originale)
COLUMNS = [
    "engine_id", "cycle",
    "setting1", "setting2", "setting3",
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21",
]

# Sensori con varianza quasi nulla su tutti i dataset: non portano info sul degrado
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
# Colonne delle condizioni operative, usate per il clustering in FD002/FD004
SETTING_COLS = ["setting1", "setting2", "setting3"]
# RUL massima: cap a 125 cicli (piecewise linear RUL, standard in letteratura)
MAX_RUL = 125


def load_raw(path: str) -> pd.DataFrame:
    # sep="\s+" gestisce separatori multipli (spazi variabili nel formato CMAPSS)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    # Rimuovo i sensori costanti: non contribuiscono alla predizione del degrado
    df.drop(columns=DROP_SENSORS, inplace=True)
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    # Calcolo il ciclo massimo per ogni motore (= fine vita)
    max_cycle = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="engine_id")
    # RUL = cicli rimanenti, cappata a MAX_RUL: nella fase iniziale (sana)
    # il motore non mostra ancora segni di degrado, quindi trattare RUL > 125
    # come 125 evita di introdurre rumore nell'apprendimento
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=MAX_RUL)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def normalize_standard(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    # Cast a float necessario: pandas 3.x rifiuta di sovrascrivere int64 con float
    train_df[feature_cols] = train_df[feature_cols].astype(float)
    test_df[feature_cols] = test_df[feature_cols].astype(float)
    # fit solo sul train → evita data leakage: il test non deve influenzare la scala
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df, scaler


def normalize_clustered(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list, n_clusters: int = 6):
    # Normalizzazione cluster-based per FD002/FD004 (6 condizioni operative):
    # con condizioni miste, un singolo scaler globale confonde regimi diversi.
    # KMeans identifica i 6 regimi operativi dai 3 settings, poi normalizziamo
    # separatamente ogni cluster → ogni regime ha la sua scala [0, 1].
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[feature_cols] = train_df[feature_cols].astype(float)
    test_df[feature_cols] = test_df[feature_cols].astype(float)
    # fit_predict sul train: apprende i centroidi e assegna ogni riga a un cluster
    train_df["cluster"] = kmeans.fit_predict(train_df[SETTING_COLS])
    # predict sul test usa i centroidi del train → no leakage
    test_df["cluster"] = kmeans.predict(test_df[SETTING_COLS])

    scalers = {}
    for c in range(n_clusters):
        scaler = MinMaxScaler()
        mask_tr = train_df["cluster"] == c
        mask_te = test_df["cluster"] == c
        # fit solo sulle righe del cluster c nel train
        train_df.loc[mask_tr, feature_cols] = scaler.fit_transform(train_df.loc[mask_tr, feature_cols])
        if mask_te.any():
            test_df.loc[mask_te, feature_cols] = scaler.transform(test_df.loc[mask_te, feature_cols])
        scalers[c] = scaler

    train_df.drop(columns=["cluster"], inplace=True)
    test_df.drop(columns=["cluster"], inplace=True)
    return train_df, test_df, scalers


def make_windows(df: pd.DataFrame, feature_cols: list, window_size: int):
    """Finestra scorrevole sulla serie temporale di ogni motore.
    Output: X di forma (n_campioni, window_size, n_feature), y scalare per ogni finestra."""
    X, y = [], []
    for _, group in df.groupby("engine_id"):
        data = group[feature_cols].values
        labels = group["rul"].values
        # Scorro la serie con passo 1: ogni finestra produce un campione di training
        # n_campioni >> n_motori grazie allo sliding window (data augmentation implicita)
        for i in range(len(data) - window_size + 1):
            X.append(data[i: i + window_size])
            # Label = RUL all'ultimo timestep della finestra (non la media)
            y.append(labels[i + window_size - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_test_windows(df: pd.DataFrame, feature_cols: list, window_size: int):
    """Per il test: estrae solo l'ultima finestra di ogni motore.
    Il file RUL_FDxxx.txt fornisce 1 etichetta per motore → 1 predizione per motore."""
    X = []
    for _, group in df.groupby("engine_id"):
        data = group[feature_cols].values
        if len(data) >= window_size:
            # Prendo gli ultimi window_size cicli: il punto più vicino al guasto
            X.append(data[-window_size:])
        else:
            # Zero-padding a sinistra se il motore ha meno cicli della finestra
            pad = np.zeros((window_size - len(data), data.shape[1]), dtype=np.float32)
            X.append(np.vstack([pad, data]))
    return np.array(X, dtype=np.float32)


def get_feature_cols(df: pd.DataFrame) -> list:
    # Restituisce tutte le colonne tranne quelle non-feature (metadati e target)
    exclude = {"engine_id", "cycle", "rul"}
    return [c for c in df.columns if c not in exclude]
