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

class ShapeLocatorNet(tf.keras.Model):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(hidden_dim // 2, 3, padding='same', activation='relu'),
            layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2D(hidden_dim // 2, 3, padding='same', activation='relu'),
            layers.Conv2D(1, 1, activation='sigmoid'),
        ])


    def call(self, x, training=False):
        # x shape: (B, H, W, T, C)
        x = x[:, :, :, -1, :]  # pega o último frame
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # shape: (B, H, W, 1)



def compile_shape_locator(model, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="shape_acc")]
    )
    return model