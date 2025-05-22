import tensorflow as tf
import tensorflow_addons as tfa
import math
NUM_CLASSES = 10

class OutputRefinement(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_classes=NUM_CLASSES, scale=0.5, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.scale = scale
        self.refine_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(hidden_dim, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(num_classes, 1)  # Output delta
        ])

    def call(self, x):
        residual = self.refine_net(x)
        return x + self.scale * residual  # Add adjustment to logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "scale": self.scale
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



class PositionalEncoding2D(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.dense = tf.keras.layers.Dense(channels, activation='tanh')

    def call(self, x):
        b, h, w, _ = tf.unstack(tf.shape(x))
        yy, xx = tf.meshgrid(tf.linspace(-1.0, 1.0, h), tf.linspace(-1.0, 1.0, w), indexing='ij')
        pos = tf.stack([yy, xx], axis=-1)
        pos = tf.expand_dims(pos, 0)
        pos = tf.tile(pos, [b, 1, 1, 1])
        encoded = self.dense(pos)
        return tf.concat([x, encoded], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FractalBlock(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
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

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EnhancedEncoder(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.blocks = [FractalBlock(dim) for _ in range(4)]
        self.out = tf.keras.layers.Conv2D(dim, 3, padding='same', activation='relu')

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out(x)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LearnedTranslation(tf.keras.layers.Layer):
    def __init__(self, max_offset=28, **kwargs):
        super().__init__(**kwargs)
        self.offset_layer = tf.keras.layers.Dense(2)
        self.max_offset = max_offset

    def call(self, x, y):  # y = output size
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        mean_feat = tf.reduce_mean(x, axis=[1, 2])  # [B, C]
        offsets = self.offset_layer(mean_feat)      # [B, 2]
        offsets = tf.clip_by_value(offsets, 0, self.max_offset)
        offsets = tf.cast(offsets, tf.int32)

        def translate(img, offset):
            dy, dx = offset[0], offset[1]
            pad_top = dy
            pad_left = dx
            pad_bottom = tf.maximum(0, y - tf.shape(img)[0] - dy)
            pad_right = tf.maximum(0, y - tf.shape(img)[1] - dx)
            paddings = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
            padded = tf.pad(img, paddings)
            padded = padded[:y, :y, :]  # crop to desired size
            return padded

        return tf.map_fn(lambda args: translate(args[0], args[1]), (x, offsets), dtype=x.dtype)


    

class LearnedFlip(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logits_layer = tf.keras.layers.Dense(4)  # [no flip, h, v, hv]

    def call(self, x):
        mean_features = tf.reduce_mean(x, axis=[1, 2])  # [B, C]
        logits = self.logits_layer(mean_features)       # [B, 4]
        flip_code = tf.argmax(logits, axis=-1)          # [B]

        def flip_op(args):
            img, code = args
            code = tf.cast(code, tf.int32)
            img = tf.identity(img)  # defensivo

            # Usamos conds porque o map_fn precisa garantir todas as branches
            return tf.case([
                (tf.equal(code, 1), lambda: tf.image.flip_left_right(img)),
                (tf.equal(code, 2), lambda: tf.image.flip_up_down(img)),
                (tf.equal(code, 3), lambda: tf.image.flip_up_down(tf.image.flip_left_right(img)))
            ], default=lambda: img)

        return tf.map_fn(flip_op, (x, flip_code), dtype=x.dtype)




class DiscreteRotation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = tf.keras.layers.Dense(4)

    def call(self, x):
        mean_features = tf.reduce_mean(x, axis=[1, 2])  # [B, C]
        logits = self.classifier(mean_features)         # [B, 4]
        k = tf.argmax(logits, axis=-1)                  # [B]

        # Rotaciona cada imagem do batch de acordo com seu valor k
        def rotate_single(args):
            img, k_val = args
            return tf.image.rot90(img, k=tf.cast(k_val, tf.int32))

        rotated = tf.map_fn(rotate_single, (x, k), dtype=x.dtype)
        return rotated
    
class LearnedColorPermutation(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        # Inicializa a matriz próxima de identidade
        initializer = tf.keras.initializers.Identity()
        self.permutation_matrix = self.add_weight(
            name='color_map',
            shape=(num_classes, num_classes),
            initializer=initializer,
            trainable=True
        )

    def call(self, x):
        # x: [B, H, W, C]
        shape = tf.shape(x)
        flat_x = tf.reshape(x, [-1, self.num_classes])  # [B*H*W, C]
        permuted = tf.matmul(flat_x, self.permutation_matrix)  # [B*H*W, C]
        return tf.reshape(permuted, shape)  # [B, H, W, C]


class LearnedRotation(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.angle_layer = tf.keras.layers.Dense(1, activation='tanh')

    def call(self, x):
        mean_features = tf.reduce_mean(x, axis=[1, 2])  # [B, C]
        angle_normed = self.angle_layer(mean_features)  # [-1, 1] range
        pi_const = tf.constant(math.pi, dtype=angle_normed.dtype)
        angle = tf.squeeze(angle_normed, axis=-1) * pi_const  # [-π, π] radians
        angle = tf.clip_by_value(angle, -pi_const, pi_const)
        rotated = tfa.image.rotate(x, angles=angle, interpolation='BILINEAR')
        return rotated

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AttentionOverMemory(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
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

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

class TemporalEncoding(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, input_channels):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_channels = input_channels
        self.linear = tf.keras.layers.Dense(hidden_dim)
        self.align = tf.keras.layers.Dense(input_channels) if hidden_dim != input_channels else None

    def call(self, x, time_index):
        time_emb = self.linear(tf.cast(time_index[:, tf.newaxis], tf.float32))  # [B, hidden]
        time_emb = tf.reshape(time_emb, [-1, 1, 1, self.hidden_dim])

        if self.align is not None:
            time_emb = self.align(time_emb)

        return x + time_emb


