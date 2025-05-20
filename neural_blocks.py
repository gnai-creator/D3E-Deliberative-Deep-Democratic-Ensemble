# neural_blocks.py

import tensorflow as tf
import tensorflow_addons as tfa
import math
NUM_CLASSES = 10

class OutputRefinement(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(num_classes, 1)
        ])

    def call(self, x):
        return self.conv(x)


class PositionalEncoding2D(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.dense = tf.keras.layers.Dense(channels, activation='tanh')

    def call(self, x):
        b, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        yy, xx = tf.meshgrid(tf.linspace(-1.0, 1.0, h), tf.linspace(-1.0, 1.0, w), indexing='ij')
        pos = tf.stack([yy, xx], axis=-1)
        pos = tf.expand_dims(pos, 0)
        pos = tf.tile(pos, [b, 1, 1, 1])
        encoded = self.dense(pos)
        return tf.concat([x, encoded], axis=-1)


class FractalBlock(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv3 = tf.keras.layers.Conv2D(dim // 2, 3, padding='same', activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(dim // 2, 5, padding='same', activation='relu')
        self.merge = tf.keras.layers.Conv2D(dim, 1, padding='same', activation=None)
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.residual = tf.keras.layers.Conv2D(dim, 1, padding='same')

    def call(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        merged = tf.concat([x3, x5], axis=-1)
        merged = self.merge(merged)
        out = self.norm(merged)
        out = self.dropout(out)
        return tf.nn.relu(out + self.residual(x))


class EnhancedEncoder(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.blocks = [FractalBlock(dim) for _ in range(4)]
        self.out = tf.keras.layers.Conv2D(dim, 3, padding='same', activation='relu')

    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x)
        return self.out(x)




class LearnedRotation(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.angle_layer = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, x):
        b = tf.shape(x)[0]
        mean_features = tf.reduce_mean(x, axis=[1, 2])  # [B, C]
        angle_normed = self.angle_layer(mean_features)  # [-1, 1] range
        angle = tf.squeeze(angle_normed, axis=-1) * math.pi  # [-π, π] radians
        angle = tf.clip_by_value(angle, -math.pi, math.pi)
        rotated = tfa.image.rotate(x, angles=angle, interpolation='BILINEAR')
        return rotated


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