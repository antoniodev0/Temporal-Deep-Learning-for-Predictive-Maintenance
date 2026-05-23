import os
import logging
import numpy as np
import tensorflow as tf

logging.getLogger("tensorflow").setLevel(logging.ERROR)
import keras
from src.metrics import evaluate


def get_callbacks(model_path: str, patience: int = 15):
    return [
        # Ferma il training se val_loss non migliora per 15 epoche consecutive;
        # restore_best_weights riporta i pesi al momento migliore osservato
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        # Salva su disco solo quando val_loss migliora → checkpoint del modello migliore
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        # Se val_loss non migliora per 7 epoche, dimezza il learning rate (min 1e-6)
        # → fine-tuning automatico: passi più piccoli quando si è vicini al minimo
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ]


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_path: str,
    epochs: int = 200,        # massimo teorico; EarlyStopping ferma prima
    batch_size: int = 256,    # buon trade-off velocità/stabilità del gradiente
    validation_split: float = 0.2,  # 80% train, 20% validation (split temporale implicito)
) -> keras.callbacks.History:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=get_callbacks(model_path),
        verbose=1,
    )
    return history


def run_experiment(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_path: str,
    **train_kwargs,
) -> dict:
    history = train_model(model, X_train, y_train, model_path, **train_kwargs)
    # Carico il checkpoint salvato da ModelCheckpoint → pesi migliori, non quelli finali
    best_model = keras.models.load_model(model_path)
    # .flatten(): predict restituisce shape (n, 1), la porto a (n,) per le metriche
    y_pred = best_model.predict(X_test, verbose=0).flatten()
    metrics = evaluate(y_test, y_pred)
    return {"metrics": metrics, "history": history.history, "predictions": y_pred}
