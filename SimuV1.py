import tensorflow as tf
from tensorflow.keras import layers

class SimuV1(tf.keras.Model):
    def __init__(self, hidden_dim=32, num_classes=10):
        super().__init__()

        # Encoder
        self.conv1 = layers.Conv3D(hidden_dim, kernel_size=3, padding='same', activation='relu')
        self.pool1 = layers.MaxPool3D(pool_size=2, padding='same')

        self.conv2 = layers.Conv3D(hidden_dim * 2, kernel_size=3, padding='same', activation='relu')
        self.pool2 = layers.MaxPool3D(pool_size=2, padding='same')

        self.conv3 = layers.Conv3D(hidden_dim * 4, kernel_size=3, padding='same', activation='relu')

        # Decoder
        self.up1 = layers.UpSampling3D(size=2)
        self.deconv1 = layers.Conv3D(hidden_dim * 2, kernel_size=3, padding='same', activation='relu')

        self.up2 = layers.UpSampling3D(size=2)
        self.deconv2 = layers.Conv3D(hidden_dim, kernel_size=3, padding='same', activation='relu')

        # Output
        self.output_layer = layers.Conv3D(num_classes, kernel_size=1, padding='same', activation='softmax')

    def call(self, x, training=False):
        assert len(x.shape) == 5, f"[SECURITY] SimuV1 recebeu input com shape inválido: {x.shape}"
        # tf.print("[SimuV1] Input shape:", tf.shape(x))

        # Encoder
        x1 = self.conv1(x)
        x2 = self.pool1(x1)

        x3 = self.conv2(x2)
        x4 = self.pool2(x3)

        x5 = self.conv3(x4)

        # Decoder
        x6 = self.up1(x5)
        x6 = self._crop_or_pad_to_match(x6, x3)
        x6 = self.deconv1(x6)

        x7 = self.up2(x6)
        x7 = self._crop_or_pad_to_match(x7, x1)
        x7 = self.deconv2(x7)

        return self.output_layer(x7)

    def _crop_or_pad_to_match(self, source, target):
        """Ajusta o shape de `source` para coincidir com `target` via crop ou padding."""
        source_shape = tf.shape(source)[1:4]
        target_shape = tf.shape(target)[1:4]

        diffs = target_shape - source_shape
        paddings = tf.maximum(diffs, 0)
        croppings = tf.maximum(-diffs, 0)

        # Aplica padding
        paddings_tf = tf.stack([[0, paddings[0]], [0, paddings[1]], [0, paddings[2]]])
        source = tf.pad(source, tf.concat([[[0, 0]], paddings_tf, [[0, 0]]], axis=0))

        # Aplica cropping
        source = source[:, :target_shape[0], :target_shape[1], :target_shape[2], :]

        return source
