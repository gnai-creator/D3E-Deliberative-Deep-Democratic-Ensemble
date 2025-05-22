import tensorflow as tf
from tensorflow.keras import layers
from neural_blocks import (
    LearnedColorPermutation,
    LearnedFlip,
    DiscreteRotation,
    PositionalEncoding2D,
    AttentionOverMemory,
    FractalBlock,

)

NUM_CLASSES = 10

class ShapeLocatorNet(tf.keras.Model):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.flip = LearnedFlip()
        self.rotation = DiscreteRotation()
        self.input_proj = layers.Conv2D(hidden_dim, 1, activation='relu')
        self.pos_enc = PositionalEncoding2D(hidden_dim)
        self.fractal = FractalBlock(hidden_dim)
        self.attn_memory = AttentionOverMemory(hidden_dim)

        self.encoder = tf.keras.Sequential([
            layers.Conv2D(hidden_dim // 2, 3, padding='same', activation='relu'),
            layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2D(hidden_dim // 2, 3, padding='same', activation='relu'),
            layers.Conv2D(NUM_CLASSES, 1)  # logits
        ])

        self.presence_head = layers.Conv2D(1, 1, activation='sigmoid')  # Presença binária
        self.color_perm = LearnedColorPermutation(NUM_CLASSES)

    def call(self, x, training=False):
        x = x[:, :, :, -1, :]
        x = self.flip(x)
        x = self.rotation(x)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.fractal(x)

        pooled = tf.reduce_mean(x, axis=[1, 2])
        memory = tf.expand_dims(pooled, axis=1)
        memory_out = self.attn_memory(memory, pooled)
        memory_out = tf.reshape(memory_out, [-1, 1, 1, x.shape[-1]])
        memory_out = tf.tile(memory_out, [1, tf.shape(x)[1], tf.shape(x)[2], 1])
        x = x + memory_out

        class_logits = self.decoder(x)
        presence_map = self.presence_head(x)

        class_logits = self.color_perm(class_logits, training=training)

        return {
            "class_logits": class_logits,       # (B, H, W, NUM_CLASSES)
            "presence_map": presence_map        # (B, H, W, 1)
        }


def compile_shape_locator(model, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="shape_acc")]
    )
    return model