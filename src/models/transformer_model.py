import numpy as np
import tensorflow as tf
import keras


# Il decorator registra la classe nel registro di Keras con un nome univoco ("rul_pred>PositionalEncoding")
# → necessario per salvare e ricaricare correttamente il modello da file .keras
@keras.saving.register_keras_serializable(package="rul_pred")
class PositionalEncoding(keras.layers.Layer):
    """Codifica posizionale sinusoidale (Vaswani et al., 2017).
    Aggiunge a ogni timestep un'impronta unica basata sulla sua posizione nella sequenza,
    senza parametri apprendibili → più stabile e con meno parametri rispetto agli embedding posizionali."""

    def __init__(self, max_len: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        d_model = input_shape[-1]  # dimensione dell'embedding
        positions = np.arange(self.max_len)[:, np.newaxis]   # shape (max_len, 1)
        dims = np.arange(d_model)[np.newaxis, :]              # shape (1, d_model)
        # Formula: angolo(pos, i) = pos / 10000^(2i / d_model)
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        # Dimensioni pari → seno, dimensioni dispari → coseno
        # Ogni posizione ha una firma unica e le posizioni vicine hanno firme simili
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        # Aggiungo dimensione batch: shape (1, max_len, d_model)
        self.pe = tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        # Sommo PE all'embedding: il modello impara a sfruttare l'informazione posizionale
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len})
        return config


@keras.saving.register_keras_serializable(package="rul_pred")
class TransformerEncoderBlock(keras.layers.Layer):
    """Blocco encoder standard del Transformer: MHA → Add&Norm → FFN → Add&Norm."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        # Multi-head attention: invece di un'unica attention su d_model dimensioni,
        # si eseguono num_heads attention parallele ciascuna su uno spazio ridotto
        # (key_dim = d_model // num_heads). Con i valori di default: 4 teste × 16 dim = 64.
        # Ogni testa può specializzarsi su un tipo diverso di relazione temporale
        # (es. dipendenze a breve termine, correlazioni a lungo raggio, pattern periodici).
        # I 4 output da 16 dim vengono concatenati → vettore da 64, stessa dim di input.
        self.attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        # FFN: espansione → ff_dim (con ReLU), poi riduzione → d_model
        # L'espansione permette di catturare interazioni non lineari tra le feature
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(d_model),
        ])
        # LayerNorm dopo ogni sotto-layer: stabilizza il training
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = keras.layers.Dropout(dropout)
        self.drop2 = keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        # Self-attention: query=key=value=x → ogni timestep "guarda" tutti gli altri
        # e apprende a pesare le dipendenze temporali rilevanti per il degrado
        attn_out = self.attn(x, x, training=training)
        # Residual connection + LayerNorm (Post-LN): x + dropout(attn) poi norma
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
    d_model: int = 64,    # dimensione interna del Transformer (embedding size)
    num_heads: int = 4,   # numero di attention head paralleli
    ff_dim: int = 128,    # dimensione FFN = 2× d_model, standard in letteratura
    num_blocks: int = 2,  # numero di blocchi encoder impilati
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
) -> keras.Model:
    inputs = keras.Input(shape=(window_size, n_features))

    # Proiezione lineare: porta i 17 sensori nello spazio d_model del Transformer
    # (i 17 sensori non hanno la dimensione giusta per MHA senza questa proiezione)
    x = keras.layers.Dense(d_model)(inputs)
    # Aggiunge l'informazione sulla posizione temporale di ogni ciclo nella finestra
    x = PositionalEncoding(max_len=window_size)(x)

    # Impilo num_blocks blocchi encoder: più profondità = più capacità rappresentativa
    for _ in range(num_blocks):
        x = TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout)(x)

    # GlobalAveragePooling1D: aggrega tutti i timestep con media → più stabile
    # rispetto a usare solo l'ultimo token (come farebbe un LSTM), perché sfrutta
    # l'intera sequenza per la predizione finale
    x = keras.layers.GlobalAveragePooling1D()(x)
    # ReLU introduce non-linearità: senza attivazione, layer densi in sequenza
    # collasserebbero in una sola trasformazione lineare, incapace di modellare il degrado
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)  # regularizzazione aggiuntiva prima dell'output
    # Uscita lineare: regressione RUL, nessuna attivazione
    outputs = keras.layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs, outputs, name="Transformer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
