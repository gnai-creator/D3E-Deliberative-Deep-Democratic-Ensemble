import tensorflow as tf
from tensorflow.keras import layers
from neural_blocks import (
    LearnedColorPermutation, #ok
    LearnedFlip, # ok
    DiscreteRotation, #ok
    PositionalEncoding2D, # ok
    AttentionOverMemory, # Ok
    FractalBlock, # Ok
    DynamicClassPermuter # OK
)

NUM_CLASSES = 10

class SimuV1(tf.keras.Model):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.flip = LearnedFlip()
        self.rotation = DiscreteRotation()
        self.input_proj = layers.Conv2D(hidden_dim, 1, activation='relu')
        self.pos_enc = PositionalEncoding2D(hidden_dim)
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(hidden_dim // 2, 3, padding='same', activation='relu'),
            layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
        ])
        self.fractal = FractalBlock(hidden_dim)
        self.attn_memory = AttentionOverMemory(hidden_dim)

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=hidden_dim)
        self.norm = tf.keras.layers.LayerNormalization()

        self.decoder = tf.keras.Sequential([
            layers.Conv2D(hidden_dim // 2, 3, padding='same', activation='relu'),
            layers.Conv2D(NUM_CLASSES, 1)  # logits finais
        ])

        self.presence_head = layers.Conv2D(1, 1, activation='sigmoid')

        # Permutadores
        self.color_perm_train = LearnedColorPermutation(NUM_CLASSES)
        self.permutation_eval = tf.range(NUM_CLASSES)  # identidade
        self.permuter = DynamicClassPermuter(num_shape_types=4, num_classes=NUM_CLASSES)

    def call(self, x, training=False):
        if x.shape.rank == 5:
            x = x[:, :, :, -1, :]
        elif x.shape.rank == 4:
            pass
        else:
            tf.print("[DEBUG] Tensor de entrada shape inesperado:", tf.shape(x))
            raise ValueError(f"[ERRO] Entrada com shape inesperado: {x.shape}")

        x = self.flip(x)
        x = self.rotation(x)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.fractal(x)

        # Memory attention
        pooled = tf.reduce_mean(x, axis=[1, 2])
        memory = tf.expand_dims(pooled, axis=1)
        memory_out = self.attn_memory(memory, pooled)
        memory_out = tf.reshape(memory_out, [-1, 1, 1, x.shape[-1]])
        memory_out = tf.tile(memory_out, [1, tf.shape(x)[1], tf.shape(x)[2], 1])
        x = x + memory_out

        # MultiHeadAttention refinamento
        b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.shape[-1]
        x_flat = tf.reshape(x, [b, h * w, c])
        x_attn = self.mha(x_flat, x_flat)
        x_attn = self.norm(x_attn + x_flat)
        x = tf.reshape(x_attn, [b, h, w, x_attn.shape[-1]])

        # Logits
        raw_logits = self.decoder(x)  # antes de qualquer permutação
        raw_logits = self.permuter(x, raw_logits)  # aplica filtro baseado no shape latente

        # Treino vs inferência
        if training:
            class_logits = self.color_perm_train(raw_logits, training=True)
            return class_logits
        else:
            class_logits = tf.gather(raw_logits, self.permutation_eval, axis=-1)
            presence_map = self.presence_head(x)
            return {
                "class_logits": class_logits,
                "presence_map": presence_map
            }


