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
    """Analisi per fascia di RUL: capisce dove il modello sbaglia di più.

    Fasce (intervalli half-open per evitare doppio conteggio ai bordi):
      - "0-50"   : [0, 50)   fase critica — motori vicini al guasto
      - "50-100" : [50, 100) fase di transizione — degrado in corso
      - "100-125": [100, 125] fase sana cappata — RUL intorno al cap di 125

    Restituisce per ogni fascia: mae, rmse, nasa_score, n (supporto statistico).
    La somma di n sulle 3 fasce è uguale al numero totale di motori nel test set.
    """
    # (lo, hi, closed_right): se closed_right=True usa <=, altrimenti <
    ranges = {
        "0-50":    (0,   50,  False),
        "50-100":  (50,  100, False),
        "100-125": (100, 125, True),
    }
    results = {}
    for label, (lo, hi, closed) in ranges.items():
        if closed:
            mask = (y_true >= lo) & (y_true <= hi)
        else:
            mask = (y_true >= lo) & (y_true < hi)
        n = int(mask.sum())
        if n == 0:
            results[label] = {"mae": None, "rmse": None, "nasa_score": None, "n": 0}
        else:
            results[label] = {
                "mae":       mae(y_true[mask], y_pred[mask]),
                "rmse":      rmse(y_true[mask], y_pred[mask]),
                "nasa_score": nasa_score(y_true[mask], y_pred[mask]),
                "n":         n,
            }
    return results
