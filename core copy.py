import tensorflow as tf
from tensorflow.keras import layers
from neural_blocks import (
    EnhancedEncoder,
    PositionalEncoding2D,
    LearnedRotation,
    AttentionOverMemory,
    OutputRefinement
)
from typing import cast, Any


NUM_CLASSES = 10

class SageAxiom(tf.keras.Model):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.pos_enc = PositionalEncoding2D(hidden_dim)
        self.early_proj = layers.Conv2D(hidden_dim, 1, activation='relu')
        self.rotation = LearnedRotation(hidden_dim)
        self.encoder = EnhancedEncoder(hidden_dim)
        self.norm = layers.LayerNormalization()

        self.agent_dense = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(0.3)
        ])
        self.memory_attention = AttentionOverMemory(hidden_dim)

        self.attn = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=hidden_dim // 4,
            dropout=0.2
        )
        self.attn_norm = layers.LayerNormalization()

        self.attn_conv = layers.Conv2D(hidden_dim, 1, activation='relu')
        self.refine_norm = layers.LayerNormalization()

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
        config.update({"hidden_dim": self.hidden_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _tile_context(self, context):
        context_tensor = tf.convert_to_tensor(context, dtype=tf.float32)
        multiples_tensor = tf.constant([1, 30, 30, 1], dtype=tf.int32)

        # Cast them to Any to silence the type checker
        return tf.tile(cast(Any, context_tensor), cast(Any, multiples_tensor))

    def call(self, x_seq, training=False):
        if x_seq.shape.rank != 4:
            raise ValueError(f"Esperado input de shape [batch, height, width, {NUM_CLASSES}]")

        batch_shape = tf.shape(x_seq)
        batch_size = tf.gather(batch_shape, 0)

        x = self.pos_enc(x_seq)
        x = self.early_proj(x)
        x = self.rotation(x)
        x_encoded = self.encoder(x, training=training)
        x_encoded = self.norm(x_encoded, training=training)

        x_flat = tf.keras.layers.GlobalAveragePooling2D()(x_encoded)
        x_flat = self.pool_dense1(x_flat)
        state = self.agent_dense(x_flat)

        memory_tensor = tf.expand_dims(state, axis=0)
        memory_context = self.memory_attention(memory_tensor, state)
        full_context = tf.concat([state, memory_context], axis=-1)

        context = tf.reshape(full_context, [batch_size, 1, 1, 2 * self.hidden_dim])
        context = self._tile_context(context)

        downsampled = tf.keras.layers.AveragePooling2D(pool_size=2)(x_encoded)
        shape = tf.shape(downsampled)
        h = tf.gather(shape, 1)
        w = tf.gather(shape, 2)
        seq_len = tf.math.multiply(h, w)

        batch_size = tf.cast(batch_size, tf.int32)
        seq_len = tf.cast(seq_len, tf.int32)
        h = tf.cast(h, tf.int32)
        w = tf.cast(w, tf.int32)

        attn_input = tf.reshape(downsampled, [batch_size, seq_len, self.hidden_dim])

        attn_output = self.attn(
            query=attn_input,
            key=attn_input,
            value=attn_input,
            training=training
        )
        attn_output = self.attn_norm(attn_output + attn_input)

        attn_output_reshaped = tf.reshape(attn_output, [batch_size, h, w, self.hidden_dim])
        attn_upsampled = tf.image.resize(attn_output_reshaped, size=[30, 30], method='bilinear')

        fused = tf.nn.relu(attn_upsampled + x_encoded)

        for _ in range(2):
            block = self.attn_conv(fused)
            block = self.refine_norm(block + fused)
            fused = tf.nn.relu(fused + block)

        logits = self.decoder(fused, training=training)
        final_logits = self.refiner(logits, training=training)

        return logits