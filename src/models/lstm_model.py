import tensorflow as tf
import keras


def build_lstm(
    window_size: int,
    n_features: int,
    units: int = 128,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> keras.Model:
    inputs = keras.Input(shape=(window_size, n_features))
    x = keras.layers.LSTM(units, return_sequences=True)(inputs)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.LSTM(units // 2)(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs, outputs, name="LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
