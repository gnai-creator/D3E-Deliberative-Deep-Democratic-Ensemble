# core.py

import tensorflow as tf
from neural_blocks import TokenEmbedding, EnhancedEncoder, PositionalEncoding2D, LearnedRotation
from neural_blocks import MultiHeadAttentionWrapper, ChoiceHypothesisModule, AttentionOverMemory, OutputRefinement
from tensorflow.keras import layers

NUM_CLASSES = 10

class SageAxiom(tf.keras.Model):
    def __init__(self, hidden_dim=64, use_hard_choice=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_hard_choice = use_hard_choice

        self.token_embedding = TokenEmbedding(vocab_size=NUM_CLASSES, dim=hidden_dim)
        self.early_proj = tf.keras.layers.Conv2D(hidden_dim, 1, activation='relu')
        self.encoder = EnhancedEncoder(hidden_dim)
        self.norm = tf.keras.layers.LayerNormalization()
        self.pos_enc = PositionalEncoding2D(hidden_dim)
        self.rotation = LearnedRotation(hidden_dim)

        self.attn = MultiHeadAttentionWrapper(hidden_dim, heads=8)
        self.agent = tf.keras.layers.GRUCell(hidden_dim, dtype="float32")

        self.chooser = ChoiceHypothesisModule(hidden_dim)
        self.attend_memory = AttentionOverMemory(hidden_dim)

        self.projector = tf.keras.layers.Conv2D(hidden_dim, 1)
        
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
        self.fallback = tf.keras.layers.Conv2D(NUM_CLASSES, 1)
        self.gate_scale = tf.keras.layers.Dense(hidden_dim, activation='tanh', name="gate_scale")

        self.refine_weight = self.add_weight(
            name="refine_weight", shape=(), initializer=tf.keras.initializers.Constant(0.5), trainable=False
        )

        self.pool_dense1 = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
            tf.keras.layers.Dropout(0.3)
        ])

    def call(self, x_seq, training=False):
        if x_seq.shape.rank != 4:
            raise ValueError(f"Esperado input de shape [batch, height, width, {NUM_CLASSES}]")

        batch = x_seq.shape[0] or tf.shape(x_seq)[0]

        xt = self.token_embedding(x_seq)
        xt = tf.ensure_shape(xt, [None, 30, 30, self.hidden_dim])
        xt = self.pos_enc(xt)
        xt = self.early_proj(xt)
        xt = self.rotation(xt)
        xt = self.encoder(xt, training=training)
        xt = self.norm(xt, training=training)

        x_flat = tf.keras.layers.GlobalAveragePooling2D()(xt)
        x_flat = self.pool_dense1(x_flat)
        x_flat = tf.cast(x_flat, tf.float32)

        state = tf.zeros([batch, self.hidden_dim], dtype=tf.float32)
        out, [state] = self.agent(x_flat, [state])

        memory_tensor = tf.expand_dims(out, axis=0)
        memory_context = self.attend_memory(memory_tensor, state)
        memory_context = tf.cast(memory_context, tf.float32)
        full_context = tf.concat([state, memory_context], axis=-1)

        context = tf.reshape(full_context, [batch, 1, 1, 2 * self.hidden_dim])
        context = tf.tile(context, [1, 30, 30, 1])
        context = tf.ensure_shape(context, [None, 30, 30, 2 * self.hidden_dim])

        projected = self.projector(context)
        attended = self.attn(projected)
        chosen = self.chooser(attended, hard=self.use_hard_choice)

        last_xt = self.token_embedding(x_seq)
        last_xt = tf.ensure_shape(last_xt, [None, 30, 30, self.hidden_dim])
        last_xt = self.pos_enc(last_xt)
        last_xt = self.early_proj(last_xt)
        last_xt = self.rotation(last_xt)
        last_xt = self.encoder(last_xt, training=training)

        channel_gate = self.gate_scale(full_context)
        channel_gate = tf.reshape(channel_gate, [batch, 1, 1, self.hidden_dim])
        channel_gate = tf.tile(channel_gate, [1, 30, 30, 1])

        blended = channel_gate * chosen + (1 - channel_gate) * last_xt
        for _ in range(2):
            refined = self.attn(blended)
            blended = tf.nn.relu(blended + refined)

        logits = self.decoder(blended, training=training)
        refined_logits = self.refiner(logits, training=training)
        conservative_logits = self.fallback(blended)

        w = tf.clip_by_value(self.refine_weight, 0.0, 1.0)
        final_logits = w * refined_logits + (1.0 - w) * conservative_logits
        return final_logits