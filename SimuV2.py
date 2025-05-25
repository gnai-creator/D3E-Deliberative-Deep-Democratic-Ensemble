import tensorflow as tf
from tensorflow.keras import layers
from neural_blocks import (
    LearnedColorPermutation,
    LearnedFlip,
    DiscreteRotation,
    PositionalEncoding2D,
    AttentionOverMemory,
    FractalBlock,
    DynamicClassPermuter,
    SpatialFocusTemporalMarking,
    ClassTemporalAlignmentBlock,
)

NUM_CLASSES = 10

class SimuV2(tf.keras.Model):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.focal_expand = SpatialFocusTemporalMarking()
        self.flip = LearnedFlip()
        self.rotation = DiscreteRotation()
        self.pos_enc = PositionalEncoding2D(hidden_dim)
        self.temporal_shape_encoder = ClassTemporalAlignmentBlock(hidden_dim)

        self.encoder = tf.keras.Sequential([
            layers.Conv2D(hidden_dim // 2, 3, padding='same', activation='relu'),
            layers.Dropout(0.25),
            layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            layers.Dropout(0.25),
        ])

        self.fractal = FractalBlock(hidden_dim)
        self.attn_memory = AttentionOverMemory(hidden_dim)

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=hidden_dim)
        self.norm = tf.keras.layers.LayerNormalization()

        self.decoder = tf.keras.Sequential([
            layers.Conv2D(hidden_dim // 2, 3, padding='same', activation='relu'),
            layers.Dropout(0.25),
            layers.Conv2D(NUM_CLASSES, 1)
        ])

        self.presence_head = layers.Conv2D(1, 1, activation='sigmoid')

        self.color_perm_train = LearnedColorPermutation(NUM_CLASSES)
        self.permutation_eval = tf.range(NUM_CLASSES)
        self.permuter = DynamicClassPermuter(num_shape_types=4, num_classes=NUM_CLASSES)

        self.fallback_dense = tf.keras.layers.Dense(10)
        self.init_conv = layers.Conv2D(self.hidden_dim, 1, activation='relu', input_shape=(None, None, 2))

    def call(self, x, training=False):
        if x.shape.rank != 5:
            tf.print("[DEBUG] Tensor de entrada shape inesperado:", tf.shape(x))
            raise ValueError(f"[ERRO] Entrada com shape inesperado: {x.shape}")
        # Entrada esperada: [Batch, Height, Width, Class, Judge]
        # Combina as dimensões de Classe e Juízo para formar o canal de entrada
        B, H, W, C, J = tf.unstack(tf.shape(x))
        x = tf.reshape(x, [B, H, W, C * J]) 
        
        features = tf.reduce_mean(x, axis=[1, 2])

        try:
            flip_logits = self.flip.logits_layer(features)
            rotation_logits = self.rotation.classifier(features)
        except Exception:
            features = self.fallback_dense(features)
            flip_logits = self.flip.logits_layer(features)
            rotation_logits = self.rotation.classifier(features)

        flip_code = tf.argmax(flip_logits, axis=-1)
        rotation_code = tf.argmax(rotation_logits, axis=-1)

        def flip_op(args):
            img, code = args
            return tf.case([
                (tf.equal(code, 1), lambda: tf.image.flip_left_right(img)),
                (tf.equal(code, 2), lambda: tf.image.flip_up_down(img)),
                (tf.equal(code, 3), lambda: tf.image.flip_up_down(tf.image.flip_left_right(img)))
            ], default=lambda: img)

        def rotate_single(args):
            img, k_val = args
            return tf.image.rot90(img, k=tf.cast(tf.squeeze(k_val), tf.int32))

        x = tf.map_fn(flip_op, (x, flip_code), fn_output_signature=x.dtype)
        x = tf.map_fn(rotate_single, (x, rotation_code), fn_output_signature=x.dtype)

        x = self.init_conv(x)
        x = self.pos_enc(x)
        x = self.focal_expand(x)
        x = self.temporal_shape_encoder(x)
        x = self.encoder(x)
        x = self.fractal(x)

        pooled = tf.reduce_mean(x, axis=[1, 2])
        memory = tf.expand_dims(pooled, axis=1)
        memory_out = self.attn_memory(memory, pooled)
        memory_out = tf.reshape(memory_out, [-1, 1, 1, x.shape[-1]])
        memory_out = tf.tile(memory_out, [1, tf.shape(x)[1], tf.shape(x)[2], 1])
        x = x + memory_out

        b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.shape[-1]
        x_flat = tf.reshape(x, [b, h * w, c])
        x_attn = self.mha(x_flat, x_flat)
        x_attn = self.norm(x_attn + x_flat)
        x = tf.reshape(x_attn, [b, h, w, x_attn.shape[-1]])

        raw_logits = self.decoder(x)
        raw_logits = self.permuter(x, raw_logits)

        if training:
            class_logits = self.color_perm_train(raw_logits, training=True)
        else:
            class_logits = tf.gather(raw_logits, self.permutation_eval, axis=-1)

        return class_logits