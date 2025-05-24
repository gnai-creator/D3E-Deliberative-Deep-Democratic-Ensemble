import tensorflow as tf
import tensorflow_addons as tfa
import math

NUM_CLASSES = 10

# Todas as camadas reestruturadas conforme o arquivo original com correções para variáveis fora do call

class PositionalEncoding2D(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.encoding_dense = tf.keras.layers.Dense(channels, activation='tanh')
        self.project_dense = tf.keras.layers.Dense(channels, activation='tanh')

    def call(self, x):
        input_shape = tf.shape(x)
        static_shape = x.shape

        if static_shape.rank == 5:
            x = x[:, :, :, 0, :]
        elif static_shape.rank == 3:
            B, T, C = input_shape[0], input_shape[1], input_shape[2]
            pos = tf.linspace(-1.0, 1.0, T)
            pos = tf.expand_dims(pos, -1)
            pos = tf.tile(pos[None, :, :], [B, 1, 1])
            encoded = self.encoding_dense(pos)
            return x + encoded
        elif static_shape.rank != 4:
            raise ValueError(f"[PositionalEncoding2D] Input rank não suportado: {static_shape.rank}")

        B, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        yy, xx = tf.meshgrid(tf.linspace(-1.0, 1.0, H), tf.linspace(-1.0, 1.0, W), indexing='ij')
        pos = tf.stack([yy, xx], axis=-1)
        pos = tf.expand_dims(pos, 0)
        pos = tf.tile(pos, [B, 1, 1, 1])

        encoded = self.encoding_dense(pos)
        if encoded.shape[-1] != tf.shape(x)[-1]:
            encoded = self.project_dense(encoded)
        encoded = tf.broadcast_to(encoded, tf.shape(x))
        return x + encoded


class FractalBlock(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.conv3 = tf.keras.layers.Conv2D(dim // 2, 3, padding='same', activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(dim // 2, 5, padding='same', activation='relu')
        self.merge = tf.keras.layers.Conv2D(dim, 1, padding='same')
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


class LearnedFlip(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logits_layer = tf.keras.layers.Dense(4)

    def call(self, x):
        mean_features = tf.reduce_mean(x, axis=[1, 2])
        logits = self.logits_layer(mean_features)
        flip_code = tf.argmax(logits, axis=-1)

        def flip_op(args):
            img, code = args
            code = tf.cast(code, tf.int32)
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
        mean_features = tf.reduce_mean(x, axis=[1, 2])
        logits = self.classifier(mean_features)
        k = tf.argmax(logits, axis=-1)

        def rotate_single(args):
            img, k_val = args
            return tf.image.rot90(img, k=tf.cast(k_val, tf.int32))

        return tf.map_fn(rotate_single, (x, k), dtype=x.dtype)


class LearnedColorPermutation(tf.keras.layers.Layer):
    def __init__(self, num_classes, reg_strength=0.01, name="learned_color_permutation"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.reg_strength = reg_strength

    def build(self, input_shape):
        self.permutation_weights = self.add_weight(
            name="perm_matrix",
            shape=(self.num_classes, self.num_classes),
            initializer=tf.keras.initializers.Identity(gain=1.0),
            trainable=True
        )

    def call(self, x, training=False):
        weights = self.permutation_weights
        output = tf.einsum('bhwc,cd->bhwd', x, weights)

        if training:
            identity = tf.eye(self.num_classes)
            reg_loss = tf.reduce_sum(tf.square(tf.nn.softmax(weights, axis=-1) - identity))
            self.add_loss(self.reg_strength * reg_loss)
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
        mean_features = tf.reduce_mean(x, axis=[1, 2])
        angle_normed = self.angle_layer(mean_features)
        pi_const = tf.constant(math.pi, dtype=angle_normed.dtype)
        angle = tf.squeeze(angle_normed, axis=-1) * pi_const
        angle = tf.clip_by_value(angle, -pi_const, pi_const)
        return tfa.image.rotate(x, angles=angle, interpolation='BILINEAR')


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


class DynamicClassPermuter(tf.keras.layers.Layer):
    def __init__(self, num_shape_types, num_classes, temperature=1.0):
        super().__init__()
        self.num_shape_types = num_shape_types
        self.num_classes = num_classes
        self.temperature = temperature

        self.shape_class_logits = self.add_weight(
            shape=(num_shape_types, num_classes),
            initializer='glorot_uniform',
            trainable=True,
            name="shape_class_logits"
        )

        self.shape_type_predictor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 1, activation='relu'),
            tf.keras.layers.Conv2D(num_shape_types, 1)
        ])

    def call(self, x, class_logits):
        shape_type_logits = self.shape_type_predictor(x)
        shape_type_probs = tf.nn.softmax(shape_type_logits / self.temperature, axis=-1)
        shape_class_mask = tf.einsum('bhws,sc->bhwc', shape_type_probs, tf.nn.sigmoid(self.shape_class_logits))
        return class_logits * shape_class_mask


class SpatialFocusTemporalMarking(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.base_kernel = tf.constant(1.0, shape=[self.kernel_size, self.kernel_size, 1, 1], dtype=tf.float32)
        self.focal_value = self.add_weight(
            name="focal_value",
            shape=[in_channels],
            initializer=tf.keras.initializers.Constant(8.0),
            trainable=True
        )

    def call(self, inputs):
        inputs_float = tf.cast(inputs, tf.float32)
        in_channels = tf.shape(inputs_float)[-1]
        fv = tf.reshape(self.focal_value, [1, 1, 1, -1])
        diff = tf.abs(inputs_float - fv)
        mask = tf.cast(diff < 0.5, tf.float32)
        kernel = tf.tile(self.base_kernel, [1, 1, in_channels, 1])
        dilated = tf.nn.depthwise_conv2d(mask, kernel, strides=[1, 1, 1, 1], padding='SAME')
        dilated = tf.minimum(dilated, 1.0)
        return inputs_float + dilated * (fv - inputs_float)


class ClassTemporalAlignmentBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(hidden_dim, return_sequences=True)
        )
        self.class_proj = tf.keras.layers.Dense(hidden_dim)

    def call(self, x):
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        x_flat = tf.reshape(x, [b, h * w, x.shape[-1]])
        x_embed = self.class_proj(x_flat)
        x_temporal = self.rnn(x_embed)
        return tf.reshape(x_temporal, [b, h, w, self.hidden_dim * 2])


class PermutationLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, input_shape):
        self.permutation = self.add_weight(
            shape=(self.num_classes,),
            initializer=tf.keras.initializers.Constant([i for i in range(self.num_classes)]),
            trainable=False
        )

    def call(self, x, logits):
        perm = tf.cast(self.permutation, tf.int32)
        return tf.gather(logits, perm, axis=-1)
