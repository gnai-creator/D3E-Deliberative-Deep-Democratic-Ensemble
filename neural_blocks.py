import tensorflow as tf
import tensorflow_addons as tfa
import math
NUM_CLASSES = 10


class PositionalEncoding2D(tf.keras.layers.Layer): #ok
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.dense = tf.keras.layers.Dense(channels, activation='tanh')

    def call(self, x):
        b, h, w, c = tf.unstack(tf.shape(x))
        yy, xx = tf.meshgrid(tf.linspace(-1.0, 1.0, h), tf.linspace(-1.0, 1.0, w), indexing='ij')
        pos = tf.stack([yy, xx], axis=-1)  # (h, w, 2)
        pos = tf.expand_dims(pos, 0)
        pos = tf.tile(pos, [b, 1, 1, 1])
        encoded = self.dense(pos)
        return x + encoded  # SOMA em vez de CONCAT


    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FractalBlock(tf.keras.layers.Layer): # ok
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




class LearnedFlip(tf.keras.layers.Layer): # ok
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




class DiscreteRotation(tf.keras.layers.Layer): # ok
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


class LearnedColorPermutation(tf.keras.layers.Layer): # ok
    def __init__(self, num_classes, reg_strength=0.01, name="learned_color_permutation"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.reg_strength = reg_strength

        initial_weights = tf.eye(num_classes) + tf.random.normal((num_classes, num_classes), stddev=0.01)
        self.permutation_weights = tf.Variable(initial_weights, trainable=True, name="perm_matrix")

    def call(self, x, training=False):
        # Usa logits diretamente
        weights = self.permutation_weights

        output = tf.einsum('bhwc,cd->bhwd', x, weights)

        if training:
            # Regularização contra identidade
            identity = tf.eye(self.num_classes)
            reg_loss = tf.reduce_sum(tf.square(tf.nn.softmax(weights, axis=-1) - identity))
            self.add_loss(self.reg_strength * reg_loss)

            # Métricas de monitoração (após softmax para interpretação humana)
            soft_weights = tf.nn.softmax(weights, axis=-1)
            self.add_metric(tf.reduce_mean(soft_weights), name="perm_mean")
            std = tf.sqrt(tf.reduce_mean(tf.square(soft_weights - tf.reduce_mean(soft_weights))))
            self.add_metric(std, name="perm_std")

        return output


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
    
class DynamicClassPermuter(tf.keras.layers.Layer):
    """
    Permutador de classe condicional baseado em shape type inferido.
    """
    def __init__(self, num_shape_types, num_classes, temperature=1.0):
        super().__init__()
        self.num_shape_types = num_shape_types
        self.num_classes = num_classes
        self.temperature = temperature

        # Logits que definem a máscara de permissão por shape
        # Shape: [S, C] — S shape types, C classes
        self.shape_class_logits = self.add_weight(
            shape=(num_shape_types, num_classes),
            initializer='glorot_uniform',
            trainable=True,
            name="shape_class_logits"
        )

        # MLP pra prever tipo de shape de cada pixel
        self.shape_type_predictor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 1, activation='relu'),
            tf.keras.layers.Conv2D(num_shape_types, 1)  # logits de tipo
        ])

    def call(self, x, class_logits):
        """
        x: features latentes [B, H, W, D]
        class_logits: predição por classe antes da máscara [B, H, W, C]
        """

        # Prever tipo de shape de cada pixel: [B, H, W, S]
        shape_type_logits = self.shape_type_predictor(x)
        shape_type_probs = tf.nn.softmax(shape_type_logits / self.temperature, axis=-1)

        # Soft mask final por pixel:
        # shape_type_probs: [B, H, W, S]
        # shape_class_logits: [S, C]
        # → shape_class_probs: [B, H, W, C]
        shape_class_mask = tf.einsum('bhws,sc->bhwc', shape_type_probs, tf.nn.sigmoid(self.shape_class_logits))

        # Aplica a máscara suavizada aos logits de classe
        masked_logits = class_logits * shape_class_mask

        return masked_logits


class TrainableFocalExpansionLayer(tf.keras.layers.Layer):
    def __init__(self, pad_value=0, **kwargs):
        super().__init__(**kwargs)
        self.pad_value = pad_value

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        # Cria kernel 3x3 para dilatar cada canal separadamente
        self.kernel = tf.constant(1.0, shape=[3, 3, self.in_channels, 1], dtype=tf.float32)
        self.focal_value = self.add_weight(
            name="focal_value",
            shape=[self.in_channels],  # Um valor por canal
            initializer=tf.keras.initializers.Constant(8.0),
            trainable=True,
            dtype=tf.float32
        )

    def call(self, inputs):
        inputs_float = tf.cast(inputs, tf.float32)

        # Gera máscara booleana suavizada por canal
        focal_value = tf.reshape(self.focal_value, [1, 1, 1, -1])
        diff = tf.abs(inputs_float - focal_value)
        mask = tf.cast(diff < 0.5, tf.float32)  # [B, H, W, C]

        # Dilata cada canal separadamente
        dilated = tf.nn.depthwise_conv2d(mask, self.kernel, strides=[1, 1, 1, 1], padding='SAME')
        dilated = tf.minimum(dilated, 1.0)

        # Cria máscara para zonas com pad_value
        pad_mask = tf.cast(tf.equal(inputs_float, self.pad_value), tf.float32)
        write_mask = dilated * pad_mask

        output = inputs_float + write_mask * (focal_value - self.pad_value)
        return output

    def get_config(self):
        return {"pad_value": self.pad_value}


class ClassTemporalAlignmentBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(hidden_dim, return_sequences=True)
        )
        self.class_proj = tf.keras.layers.Dense(hidden_dim)

    def call(self, x):
        # x: [B, H, W, C] — one-hot ou logits

        b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.shape[3]

        # Passo 1: Flatten spatial info
        x_flat = tf.reshape(x, [b, h * w, c])  # [B, S, C]

        # Passo 2: Projeta classes para embedding space
        x_embed = self.class_proj(x_flat)  # [B, S, D]

        # Passo 3: Roda RNN sobre a sequência
        x_temporal = self.rnn(x_embed)  # [B, S, D]

        # Passo 4: Reprojeta de volta pro espaço original
        x_out = tf.reshape(x_temporal, [b, h, w, -1])

        return x_out
