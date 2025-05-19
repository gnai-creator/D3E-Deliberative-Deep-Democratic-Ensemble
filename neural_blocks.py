# neural_blocks.py

import tensorflow as tf

class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size=10, dim=64):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=dim, name="token_embedding")

    def call(self, x):
        if x.shape.rank != 4:
            raise ValueError(f"TokenEmbedding espera input 4D [B, H, W, C], recebeu: {x.shape}")
        is_onehot = tf.equal(tf.shape(x)[-1], self.embedding.input_dim)
        indices = tf.cond(is_onehot,
                          lambda: tf.argmax(x, axis=-1, output_type=tf.int32),
                          lambda: tf.cast(x, tf.int32))
        shape = tf.shape(indices)
        flat = tf.reshape(indices, [-1])
        embedded = self.embedding(flat)
        return tf.reshape(embedded, [shape[0], shape[1], shape[2], -1])


class OutputRefinement(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu')
        self.output_layer = tf.keras.layers.Conv2D(10, 1)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.output_layer(x)


class PositionalEncoding2D(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.dense = tf.keras.layers.Dense(channels, activation='tanh')

    def call(self, x):
        b, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        y = tf.linspace(-1.0, 1.0, h)
        x_pos = tf.linspace(-1.0, 1.0, w)
        yy, xx = tf.meshgrid(y, x_pos, indexing='ij')
        pos = tf.stack([yy, xx], axis=-1)
        pos = tf.expand_dims(pos, 0)
        pos = tf.tile(pos, [b, 1, 1, 1])
        encoded = self.dense(pos)
        return tf.concat([x, encoded], axis=-1)


class FractalEncoder(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv3 = tf.keras.layers.Conv2D(dim // 2, 3, padding='same', activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(dim // 2, 5, padding='same', activation='relu')
        self.merge = tf.keras.layers.Conv2D(dim, 1, padding='same', activation='relu')
        self.res = tf.keras.layers.Conv2D(dim, 1, padding='same')

    def call(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        merged = tf.concat([x3, x5], axis=-1)
        return tf.nn.relu(self.merge(merged) + self.res(x))


class FractalBlock(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(dim, 3, padding='same', activation='relu')
        self.bn = tf.keras.layers.BatchNormalization()
        self.skip = tf.keras.layers.Conv2D(dim, 1, padding='same')

    def call(self, x, training=False):
        y = self.conv(x)
        y = self.bn(y, training=training)
        return tf.nn.relu(y + self.skip(x))


class MultiHeadAttentionWrapper(tf.keras.layers.Layer):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads)

    def call(self, x):
        return self.attn(query=x, value=x, key=x)


class LearnedRotation(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.selector = tf.keras.layers.Dense(4, activation='softmax')

    def call(self, x):
        b = tf.shape(x)[0]
        pooled = tf.reduce_mean(x, axis=[1, 2])
        weights = tf.reshape(self.selector(pooled), [b, 4, 1, 1, 1])
        rotations = [x] + [tf.image.rot90(x, k=i) for i in range(1, 4)]
        stacked = tf.stack(rotations, axis=1)
        return tf.reduce_sum(stacked * weights, axis=1)


class ChoiceHypothesisModule(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.input_proj = tf.keras.layers.Conv2D(dim, 1, activation='relu')
        self.h_layers = [tf.keras.layers.Conv2D(dim, 1, activation='relu') for _ in range(4)]
        self.meander = tf.keras.Sequential([
            tf.keras.layers.Conv2D(dim, 1, activation='relu'),
            tf.keras.layers.Conv2D(dim, 1)
        ])
        self.selector = tf.keras.layers.Dense(5, activation='softmax')

    def call(self, x, hard=False):
        x = self.input_proj(x)
        candidates = [layer(x) for layer in self.h_layers] + [self.meander(x)]
        stacked = tf.stack(candidates, axis=1)
        weights = tf.reshape(self.selector(tf.reduce_mean(x, axis=[1, 2])), [-1, 5, 1, 1, 1])

        if hard:
            idx = tf.argmax(tf.squeeze(weights, axis=[2, 3, 4]), axis=-1)
            one_hot = tf.one_hot(idx, depth=5)[:, :, None, None, None]
            return tf.reduce_sum(stacked * one_hot, axis=1)
        return tf.reduce_sum(stacked * weights, axis=1)


class AttentionOverMemory(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = tf.keras.layers.Dense(dim)
        self.k_proj = tf.keras.layers.Dense(dim)
        self.v_proj = tf.keras.layers.Dense(dim)

    def call(self, memory, query):
        q = self.q_proj(query)[:, None, :]
        k = self.k_proj(memory)
        v = self.v_proj(memory)
        scores = tf.reduce_sum(q * k, axis=-1, keepdims=True)
        attn = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(attn * v, axis=1)


class EnhancedEncoder(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.blocks = [
            FractalEncoder(dim),
            FractalBlock(dim),
            FractalBlock(dim),
            FractalBlock(dim)
        ]
        self.output_layer = tf.keras.layers.Conv2D(dim, 3, padding='same', activation='relu')

    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training) if hasattr(block, 'call') and 'training' in block.call.__code__.co_varnames else block(x)
        return self.output_layer(x)
