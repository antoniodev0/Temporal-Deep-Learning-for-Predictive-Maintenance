import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

COLUMNS = [
    "engine_id", "cycle",
    "setting1", "setting2", "setting3",
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21",
]

DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
SETTING_COLS = ["setting1", "setting2", "setting3"]
MAX_RUL = 125


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df.drop(columns=DROP_SENSORS, inplace=True)
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="engine_id")
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=MAX_RUL)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def normalize_standard(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[feature_cols] = train_df[feature_cols].astype(float)
    test_df[feature_cols] = test_df[feature_cols].astype(float)
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, test_df, scaler


def normalize_clustered(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list, n_clusters: int = 6):
    """Cluster-based normalization for multi-condition sub-datasets (FD002, FD004)."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[feature_cols] = train_df[feature_cols].astype(float)
    test_df[feature_cols] = test_df[feature_cols].astype(float)
    train_df["cluster"] = kmeans.fit_predict(train_df[SETTING_COLS])
    test_df["cluster"] = kmeans.predict(test_df[SETTING_COLS])

    scalers = {}
    for c in range(n_clusters):
        scaler = MinMaxScaler()
        mask_tr = train_df["cluster"] == c
        mask_te = test_df["cluster"] == c
        train_df.loc[mask_tr, feature_cols] = scaler.fit_transform(train_df.loc[mask_tr, feature_cols])
        if mask_te.any():
            test_df.loc[mask_te, feature_cols] = scaler.transform(test_df.loc[mask_te, feature_cols])
        scalers[c] = scaler

    train_df.drop(columns=["cluster"], inplace=True)
    test_df.drop(columns=["cluster"], inplace=True)
    return train_df, test_df, scalers


def make_windows(df: pd.DataFrame, feature_cols: list, window_size: int):
    """Slide a window over each engine's time series. Returns X (n, w, f) and y (n,)."""
    X, y = [], []
    for _, group in df.groupby("engine_id"):
        data = group[feature_cols].values
        labels = group["rul"].values
        for i in range(len(data) - window_size + 1):
            X.append(data[i: i + window_size])
            y.append(labels[i + window_size - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def make_test_windows(df: pd.DataFrame, feature_cols: list, window_size: int):
    """Last window of each engine for test evaluation."""
    X = []
    for _, group in df.groupby("engine_id"):
        data = group[feature_cols].values
        if len(data) >= window_size:
            X.append(data[-window_size:])
        else:
            pad = np.zeros((window_size - len(data), data.shape[1]), dtype=np.float32)
            X.append(np.vstack([pad, data]))
    return np.array(X, dtype=np.float32)


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"engine_id", "cycle", "rul"}
    return [c for c in df.columns if c not in exclude]
