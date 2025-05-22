import tensorflow as tf
from tensorflow.keras import layers
from neural_blocks import (
    EnhancedEncoder,
    PositionalEncoding2D,
    LearnedTranslation,
    DiscreteRotation,
    LearnedFlip,
    LearnedColorPermutation,
    AttentionOverMemory,
    TemporalEncoding,
    OutputRefinement
)

NUM_CLASSES = 10

class SageUNet(tf.keras.Model):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.temporal_enc = TemporalEncoding(hidden_dim, input_channels=10)
        self.pos_enc = PositionalEncoding2D(hidden_dim)
        self.early_proj = layers.Conv2D(hidden_dim, 1, activation='relu')
        self.rotation = DiscreteRotation()
        self.flip =  LearnedFlip()
        self.encoder = EnhancedEncoder(hidden_dim)
        self.encoder_norm = layers.LayerNormalization()
        self.translation = LearnedTranslation(max_offset=30)  # ou 30 se quiser agressivo

        self.down_conv = layers.Conv2D(hidden_dim, 3, strides=2, padding='same', activation='relu')
        self.skip_conv = layers.Conv2D(hidden_dim, 1, activation='relu')

        self.attn = layers.MultiHeadAttention(num_heads=4, key_dim=hidden_dim // 4, dropout=0.2)
        self.attn_norm = layers.LayerNormalization()

        self.upsample_conv = layers.Conv2DTranspose(hidden_dim, 3, strides=2, padding='same', activation='relu')

        self.decoder = tf.keras.Sequential([
            layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(hidden_dim, 1, activation='relu'),
            layers.Conv2D(NUM_CLASSES, 1)  # logits
        ])

        self.refiner = OutputRefinement(hidden_dim, num_classes=NUM_CLASSES)
        self.color_transposer = LearnedColorPermutation(NUM_CLASSES)
        self.agent_dense = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(0.3)
        ])
        self.memory_attention = AttentionOverMemory(hidden_dim)
        self.pool_dense1 = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(0.3)
        ])
        self.context_proj = layers.Conv2D(hidden_dim, 1, activation='relu')

    def entropy(self, logits):
        probs = tf.nn.softmax(logits, axis=-1)
        ent = -tf.reduce_sum(probs * tf.math.log(probs + 1e-9), axis=-1, keepdims=True)  # [B, H, W, 1]
        return ent

    def call(self, x_seq, training=False):
        batch_size = tf.shape(x_seq)[0]
        time_steps = tf.shape(x_seq)[3]

        frames = tf.TensorArray(dtype=tf.float32, size=time_steps)

        for t in tf.range(time_steps):
            x_t = x_seq[:, :, :, t, :]
            t_tensor = tf.broadcast_to(tf.cast(t, tf.int32), [batch_size])
            x_t = self.temporal_enc(x_t, t_tensor)
            x_t = x_t + tf.random.normal(tf.shape(x_t), stddev=0.025)  # Add noise

            x_t = self.pos_enc(x_t)
            x_t = self.early_proj(x_t)
            x_t = self.rotation(x_t)
            x_t = self.flip(x_t)
            x_t = self.encoder(x_t, training=training)
            x_t = self.encoder_norm(x_t, training=training)
            frames = frames.write(t, x_t)

        x_encoded = tf.transpose(frames.stack(), perm=[1, 0, 2, 3, 4])  # [B, T, H, W, C]
        x_encoded_avg = tf.reduce_mean(x_encoded, axis=1)  # [B, H, W, C]
        x_encoded_avg = self.translation(x_encoded_avg, y=30)  # [B, H_out, W_out, C]
        # logits = self.decoder(x_encoded_avg)  
        # noise_scale = tf.stop_gradient(self.entropy(logits))  # shape: [B, H, W, 1
        # x_encoded_avg += tf.random.normal(tf.shape(x_encoded_avg)) * noise_scale

        # Context encoding
        x_flat = tf.keras.layers.GlobalAveragePooling2D()(x_encoded_avg)
        x_flat = self.pool_dense1(x_flat)
        state = self.agent_dense(x_flat)

        memory_tensor = tf.expand_dims(state, axis=0)
        memory_context = self.memory_attention(memory_tensor, state)
        full_context = tf.concat([state, memory_context], axis=-1)
        context = tf.reshape(full_context, [batch_size, 1, 1, 2 * self.hidden_dim])
        context = tf.tile(context, [1, tf.shape(x_encoded_avg)[1], tf.shape(x_encoded_avg)[2], 1])
        context = self.context_proj(context)  # reduce to hidden_dim

        # Downsample path with attention
        skip_connection = self.skip_conv(x_encoded_avg)
        downsampled = self.down_conv(x_encoded_avg)
        h, w = tf.shape(downsampled)[1], tf.shape(downsampled)[2]
        seq_len = h * w

        attn_input = tf.reshape(downsampled, [batch_size, seq_len, self.hidden_dim])
        attn_output = self.attn(query=attn_input, key=attn_input, value=attn_input, training=training)
        attn_output = self.attn_norm(attn_output + attn_input)
        attn_output_reshaped = tf.reshape(attn_output, [batch_size, h, w, self.hidden_dim])

        # Upsample + skip connection
        upsampled = self.upsample_conv(attn_output_reshaped)

        # Ensure compatibility of tensor shapes before addition
        skip_shape = tf.shape(skip_connection)[-1]
        context_shape = tf.shape(context)[-1]
        upsampled_shape = tf.shape(upsampled)[-1]
        min_channels = tf.reduce_min([skip_shape, context_shape, upsampled_shape])

        skip_connection = skip_connection[:, :, :, :min_channels]
        context = context[:, :, :, :min_channels]
        upsampled = upsampled[:, :, :, :min_channels]

        fused = tf.nn.relu(upsampled + skip_connection + context)

        color_logits = self.decoder(fused, training=training)
        position_logits = self.refiner(color_logits, training=training)
        transposed_logits = self.color_transposer(color_logits, training=training)
        position_logits = self.refiner(transposed_logits, training=training)

        return position_logits, color_logits