import tensorflow as tf
from tensorflow.keras import layers
from neural_blocks import (
    EnhancedEncoder,
    PositionalEncoding2D,
    LearnedRotation,
    AttentionOverMemory,
    OutputRefinement
)

NUM_CLASSES = 10

class SageAxiomD(tf.keras.Model):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.pos_enc = PositionalEncoding2D(hidden_dim)
        self.early_proj = layers.Conv2D(hidden_dim, 1, activation='relu')
        self.rotation = LearnedRotation(hidden_dim)
        self.encoder = EnhancedEncoder(hidden_dim)
        self.norm = layers.LayerNormalization()

        self.decoder = tf.keras.Sequential([
            layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(hidden_dim, 1, activation='relu'),
            layers.Conv2D(NUM_CLASSES, 1)
        ])

        self.refiner = OutputRefinement(hidden_dim, num_classes=NUM_CLASSES)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x_seq, training=False):
        if x_seq.shape.rank != 4:
            raise ValueError(f"Esperado input de shape [batch, height, width, {NUM_CLASSES}]")

        x = self.pos_enc(x_seq)
        x = self.early_proj(x)
        x = self.rotation(x)
        x_encoded = self.encoder(x, training=training)
        x_encoded = self.norm(x_encoded, training=training)

        logits = self.decoder(x_encoded, training=training)
        final_logits = self.refiner(logits, training=training)

        return final_logits