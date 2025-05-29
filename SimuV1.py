import tensorflow as tf

class SimuV1(tf.keras.Model):
    def __init__(self, hidden_dim=32):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv3D(hidden_dim, kernel_size=3, padding="same", activation="relu")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.drop1 = tf.keras.layers.Dropout(0.1)
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding="same")

        self.conv2 = tf.keras.layers.Conv3D(hidden_dim * 2, kernel_size=3, padding="same", activation="relu")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.drop2 = tf.keras.layers.Dropout(0.1)
        self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 1), padding="same")

        self.conv3 = tf.keras.layers.Conv3D(hidden_dim * 4, kernel_size=3, padding="same", activation="relu")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.drop3 = tf.keras.layers.Dropout(0.1)

        self.up1 = tf.keras.layers.UpSampling3D(size=(2, 2, 1))
        self.conv4 = tf.keras.layers.Conv3D(hidden_dim * 2, kernel_size=3, padding="same", activation="relu")
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.drop4 = tf.keras.layers.Dropout(0.1)

        self.up2 = tf.keras.layers.UpSampling3D(size=(2, 2, 1))
        self.conv5 = tf.keras.layers.Conv3D(hidden_dim, kernel_size=3, padding="same", activation="relu")
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.drop5 = tf.keras.layers.Dropout(0.1)

        self.crop = tf.keras.layers.Cropping3D(((1, 1), (1, 1), (0, 0)))  # Para garantir 30x30 após upsampling

        self.conv_reduce_depth = tf.keras.layers.Conv3D(hidden_dim, kernel_size=(1, 1, 1), padding="same", activation="relu")

        self.output_layer = tf.keras.layers.Conv3D(1, kernel_size=1, activation=None, dtype='float32')

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.drop3(x)

        x = self.up1(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.drop4(x)

        x = self.up2(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.drop5(x)

        x = self.crop(x)  # Corrige para shape final 30x30

        x = self.conv_reduce_depth(x)
        x = self.output_layer(x)  # Saída: (1, 30, 30, 1, 1)

        # Converte valores contínuos para inteiros próximos
        x = tf.cast(x, tf.float32)  # Garante dtype compatível
        x = tf.round(x)             # Arredonda para valores inteiros
        x = tf.clip_by_value(x, 0, 9)    # Garante que não saia do intervalo
        return x
