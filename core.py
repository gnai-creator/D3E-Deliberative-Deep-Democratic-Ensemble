# refactored_core.py

import tensorflow as tf
from tensorflow.keras import layers
from neural_blocks import (
    EnhancedEncoder,
    PositionalEncoding2D,
    LearnedRotation,
    MultiHeadAttentionWrapper,
    ChoiceHypothesisModule,
    AttentionOverMemory,
    OutputRefinement
)

NUM_CLASSES = 10

class SageAxiom(tf.keras.Model):
    def __init__(self, hidden_dim=64, use_hard_choice=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_hard_choice = use_hard_choice

        # Embedding and preprocessing
        self.pos_enc = PositionalEncoding2D(hidden_dim)
        self.early_proj = layers.Conv2D(hidden_dim, 1, activation='relu')
        self.rotation = LearnedRotation(hidden_dim)

        # Encoder
        self.encoder = EnhancedEncoder(hidden_dim)
        self.norm = layers.LayerNormalization()

        # Attention and reasoning modules
        self.attn = MultiHeadAttentionWrapper(hidden_dim, heads=8)
        self.agent = layers.GRUCell(hidden_dim, dtype="float32")
        self.memory_attention = AttentionOverMemory(hidden_dim)
        self.hypothesis_selector = ChoiceHypothesisModule(hidden_dim)

        # Decoder
        self.projector = layers.Conv2D(hidden_dim, 1)
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
        self.fallback = layers.Conv2D(NUM_CLASSES, 1)

        # Gating mechanism
        self.channel_gate = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='tanh'),
            layers.Reshape((1, 1, hidden_dim))
        ])

        # Output blend weight (not trainable)
        self.refine_weight = self.add_weight(
            name="refine_weight",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=False
        )

        # Pooling
        self.pool_dense1 = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(0.3)
        ])

        self.last_attention_output = None

    def call(self, x_seq, training=False):
        if x_seq.shape.rank != 4:
            raise ValueError(f"Esperado input de shape [batch, height, width, {NUM_CLASSES}]")

        batch = tf.shape(x_seq)[0]

        # === Feature extraction ===
        xt = self.pos_enc(x_seq)
        xt = self.early_proj(xt)
        xt = self.rotation(xt)
        xt = self.encoder(xt, training=training)
        xt = self.norm(xt, training=training)

        x_flat = tf.keras.layers.GlobalAveragePooling2D()(xt)
        x_flat = self.pool_dense1(x_flat)
        x_flat = tf.cast(x_flat, tf.float32)

        # === Recurrent agent reasoning ===
        state = tf.zeros([batch, self.hidden_dim], dtype=tf.float32)
        out, [state] = self.agent(x_flat, [state])

        memory_tensor = tf.expand_dims(out, axis=0)
        memory_context = self.memory_attention(memory_tensor, state)
        memory_context = tf.cast(memory_context, tf.float32)

        full_context = tf.concat([state, memory_context], axis=-1)
        context = tf.reshape(full_context, [batch, 1, 1, 2 * self.hidden_dim])
        context = tf.tile(context, [1, 30, 30, 1])

        # === Attention projection ===
        projected = self.projector(context)
        attended = self.attn(projected)
        self.last_attention_output = attended

        chosen = self.hypothesis_selector(attended, hard=self.use_hard_choice)

        # === Feature gating and fusion ===
        channel_gate = self.channel_gate(full_context)
        channel_gate = tf.tile(channel_gate, [1, 30, 30, 1])

        blended = channel_gate * chosen + (1 - channel_gate) * xt

        for _ in range(2):
            refined = self.attn(blended)
            blended = tf.nn.relu(blended + refined)

        # === Decoding and refinement ===
        logits = self.decoder(blended, training=training)
        refined_logits = self.refiner(logits, training=training)
        conservative_logits = self.fallback(blended)

        w = tf.clip_by_value(self.refine_weight, 0.0, 1.0)
        final_logits = w * refined_logits + (1.0 - w) * conservative_logits

        return final_logits