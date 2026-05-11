import numpy as np
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="rul_pred")
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, max_len: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        d_model = input_shape[-1]
        positions = np.arange(self.max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len})
        return config


@keras.saving.register_keras_serializable(package="rul_pred")
class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(d_model),
        ])
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = keras.layers.Dropout(dropout)
        self.drop2 = keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.attn(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out, training=training))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.attn.key_dim * self.attn._num_heads,
            "num_heads": self.attn._num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "dropout": self.drop1.rate,
        })
        return config


def build_transformer(
    window_size: int,
    n_features: int,
    d_model: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_blocks: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
) -> keras.Model:
    inputs = keras.Input(shape=(window_size, n_features))
    x = keras.layers.Dense(d_model)(inputs)
    x = PositionalEncoding(max_len=window_size)(x)
    for _ in range(num_blocks):
        x = TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs, outputs, name="Transformer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
