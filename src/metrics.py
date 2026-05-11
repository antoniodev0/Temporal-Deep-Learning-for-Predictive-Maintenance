import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric NASA scoring function. Lower is better (0 = perfect)."""
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(score))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }


def evaluate_by_rul_range(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ranges = {"0-50": (0, 50), "50-100": (50, 100), "100-125": (100, 125)}
    results = {}
    for label, (lo, hi) in ranges.items():
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
