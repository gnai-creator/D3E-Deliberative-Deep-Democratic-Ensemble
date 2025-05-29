import tensorflow as tf
from tensorflow.keras import layers, Model

class SimuV1(Model):
    def __init__(self, hidden_dim=16, output_channels=1):
        super(SimuV1, self).__init__()

        self.conv1 = layers.Conv3D(hidden_dim, (3, 3, 3), padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.drop1 = layers.Dropout(0.2)
        self.pool1 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')

        self.conv2 = layers.Conv3D(hidden_dim * 2, (3, 3, 3), padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.drop2 = layers.Dropout(0.3)
        self.pool2 = layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same')

        self.conv3 = layers.Conv3D(hidden_dim * 4, (3, 3, 3), padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.drop3 = layers.Dropout(0.4)

        self.up1 = layers.UpSampling3D(size=(1, 2, 2))
        self.conv4 = layers.Conv3D(hidden_dim * 2, (3, 3, 3), padding='same', activation='relu')
        self.bn4 = layers.BatchNormalization()
        self.drop4 = layers.Dropout(0.3)

        self.up2 = layers.UpSampling3D(size=(1, 2, 2))
        self.conv5 = layers.Conv3D(hidden_dim, (3, 3, 3), padding='same', activation='relu')
        self.bn5 = layers.BatchNormalization()
        self.drop5 = layers.Dropout(0.2)

        self.crop = layers.Cropping3D(cropping=((0, 0), (2, 2), (2, 2)))
        self.conv_reduce_depth = tf.keras.layers.Conv3D(hidden_dim, kernel_size=(1, 1, 1), padding="same", activation="relu")
        self.output_layer = layers.Conv3D(output_channels, (1, 1, 1), activation='linear')

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.drop3(x, training=training)

        x = self.up1(x)
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.drop4(x, training=training)

        x = self.up2(x)
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.drop5(x, training=training)

        # x = self.crop(x)
        x = self.conv_reduce_depth(x)
        x = self.output_layer(x)
        # Ajuste de shape para (30, 30)
        x = x[:, :30, :30, :1, :1]
        # tf.debugging.check_numerics(x, "NaN ou Inf detectado após output_layer")
        return x