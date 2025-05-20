# core.py

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

class SageAxiomA(tf.keras.Model):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.pos_enc = PositionalEncoding2D(hidden_dim)
        self.early_proj = layers.Conv2D(hidden_dim, 1, activation='relu')
        self.rotation = LearnedRotation(hidden_dim)
        self.encoder = EnhancedEncoder(hidden_dim)
        self.norm = layers.LayerNormalization()

        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=self.hidden_dim // 4,
            dropout=0.2
        )
        self.attn_norm = layers.LayerNormalization()

        self.agent_dense = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(0.3)
        ])

        self.memory_attention = AttentionOverMemory(hidden_dim)

        self.projector = tf.keras.Sequential([
            layers.Conv2D(hidden_dim, 1, input_shape=(30, 30, 2 * hidden_dim), activation='relu'),
            layers.ReLU()
        ])

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

        self.pool_dense1 = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(0.3)
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            # adicione os parâmetros que deseja serializar
            "hidden_dim": self.hidden_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x_seq, training=False):
        if x_seq.shape.rank != 4:
            raise ValueError(f"Esperado input de shape [batch, height, width, {NUM_CLASSES}]")

        batch = tf.shape(x_seq)[0]

        x = self.pos_enc(x_seq)
        x = self.early_proj(x)
        x = self.rotation(x)
        x_encoded = self.encoder(x, training=training)
        x_encoded = self.norm(x_encoded, training=training)

        x_flat = tf.keras.layers.GlobalAveragePooling2D()(x_encoded)
        x_flat = self.pool_dense1(x_flat)
        x_flat = tf.cast(x_flat, tf.float32)

        state = self.agent_dense(x_flat)

        memory_tensor = tf.expand_dims(state, axis=0)
        memory_context = self.memory_attention(memory_tensor, state)
        memory_context = tf.cast(memory_context, tf.float32)
        full_context = tf.concat([state, memory_context], axis=-1)

        context = tf.reshape(full_context, [batch, 1, 1, 2 * self.hidden_dim])
        context = tf.tile(context, [1, 30, 30, 1])

        projected = self.projector(context)
        attended = self.attn(
            query=projected,
            value=projected,
            key=projected,
            training=training
        )
        attended = self.attn_norm(attended + projected)

        fused = tf.nn.relu(attended + x_encoded)

        for _ in range(2):
            refined = self.attn(query=fused, value=fused, key=fused, training=training)
            refined = self.attn_norm(refined + fused)
            fused = tf.nn.relu(fused + refined)

        logits = self.decoder(fused, training=training)
        final_logits = self.refiner(logits, training=training)

        return final_logits
