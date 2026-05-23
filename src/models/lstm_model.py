import tensorflow as tf
import keras


def build_lstm(
    window_size: int,
    n_features: int,
    units: int = 128,      # neuroni nel primo layer LSTM
    dropout: float = 0.2,  # probabilità di azzeramento casuale → regularizzazione
    learning_rate: float = 1e-3,
) -> keras.Model:
    inputs = keras.Input(shape=(window_size, n_features))

    # Primo LSTM: return_sequences=True → restituisce l'output per ogni timestep,
    # necessario per alimentare il secondo LSTM (che riceve una sequenza, non un vettore)
    x = keras.layers.LSTM(units, return_sequences=True)(inputs)
    x = keras.layers.Dropout(dropout)(x)

    # Secondo LSTM con units//2: comprime la rappresentazione temporale,
    # distillando le feature più rilevanti per la predizione finale
    # return_sequences=False (default) → restituisce solo l'ultimo hidden state
    x = keras.layers.LSTM(units // 2)(x)
    x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Dense(64, activation="relu")(x)
    # Uscita lineare (nessuna attivazione): task di regressione, RUL può essere
    # qualsiasi valore positivo → non vincoliamo l'output
    outputs = keras.layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs, outputs, name="LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        # MSE come loss: penalizza gli errori grandi quadraticamente,
        # coerente con RMSE come metrica di valutazione principale
        loss="mse",
        metrics=["mae"],
    )
    return model
