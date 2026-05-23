import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Penalizza gli errori grandi in modo quadratico: sensibile agli outlier
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Errore medio assoluto: più robusto agli outlier, più interpretabile del RMSE
    return float(np.mean(np.abs(y_true - y_pred)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Funzione di scoring asimmetrica NASA. Punteggio più basso = migliore (0 = perfetto)."""
    # d > 0: predizione tardiva (sovrastima RUL) → il motore è più vicino al guasto di quanto pensiamo
    # d < 0: predizione anticipata (sottostima RUL) → interveniamo prima del necessario
    d = y_pred - y_true
    # Penalità asimmetrica: exp(d/10) per predizioni tardive cresce più veloce
    # di exp(-d/13) per quelle anticipate → il guasto non previsto costa di più
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    # sum (non mean): riflette il rischio cumulativo sull'intera flotta di motori
    return float(np.sum(score))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }


def evaluate_by_rul_range(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Analisi per fascia di RUL: capisce dove il modello sbaglia di più
    # 0-50: fase critica (vicini al guasto), 50-100: transizione, 100-125: fase sana cappata
    ranges = {"0-50": (0, 50), "50-100": (50, 100), "100-125": (100, 125)}
    results = {}
    for label, (lo, hi) in ranges.items():
        # Maschera booleana per isolare i campioni nella fascia [lo, hi]
        mask = (y_true >= lo) & (y_true <= hi)
        if mask.sum() == 0:
            results[label] = {"mae": None, "rmse": None, "n": 0}
        else:
            results[label] = {
                "mae": mae(y_true[mask], y_pred[mask]),
                "rmse": rmse(y_true[mask], y_pred[mask]),
                "n": int(mask.sum()),
            }
    return results
