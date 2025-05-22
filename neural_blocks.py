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
    
class ColorTransposer(tf.keras.layers.Layer):
    def __init__(self, num_classes, use_softmax=True, reg_strength=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.use_softmax = use_softmax
        self.reg_strength = reg_strength

        initial_weights = tf.eye(num_classes) + tf.random.normal(
            (num_classes, num_classes), stddev=0.01
        )
        self.permutation_weights = tf.Variable(initial_weights, trainable=True)

    def call(self, x):
        weights = self.permutation_weights
        if self.use_softmax:
            weights = tf.nn.softmax(weights, axis=-1)

        # Regularização L2 para manter próximo da identidade
        identity = tf.eye(self.num_classes)
        reg_loss = tf.reduce_sum(tf.square(weights - identity))
        self.add_loss(self.reg_strength * reg_loss)

        return tf.stop_gradient(tf.einsum('bhwc,cd->bhwd', x, weights)) + 0.0 * tf.einsum('bhwc,cd->bhwd', x, weights)




class LearnedColorPermutation(tf.keras.layers.Layer):
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


class ClassEmbeddingMatcher(tf.keras.layers.Layer):
    """
    Transforma representações contínuas em predições discretas de classe
    via similaridade com vetores de classe (embeddings).

    - Entrada: Tensor contínuo [B, H, W, D]
    - Saída: Logits [B, H, W, C], onde C = num_classes
    """
    def __init__(self, num_classes, embedding_dim, use_temperature=True):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_temperature = use_temperature
        self.class_embeddings = self.add_weight(
            shape=(num_classes, embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="class_embeddings"
        )
        if self.use_temperature:
            self.temperature = self.add_weight(
                shape=(),
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True,
                name="temperature"
            )

    def call(self, x):
        # x: [B, H, W, D]
        # Reshape para [B*H*W, D]
        orig_shape = tf.shape(x)
        flat_x = tf.reshape(x, [-1, self.embedding_dim])  # [N, D]

        # Normalize embeddings and inputs
        norm_x = tf.nn.l2_normalize(flat_x, axis=-1)  # [N, D]
        norm_classes = tf.nn.l2_normalize(self.class_embeddings, axis=-1)  # [C, D]

        # Similaridade coseno: [N, C]
        logits = tf.matmul(norm_x, norm_classes, transpose_b=True)

        if self.use_temperature:
            logits = logits / self.temperature

        # Volta pro shape [B, H, W, C]
        new_shape = tf.concat([orig_shape[:-1], [self.num_classes]], axis=0)
        logits = tf.reshape(logits, new_shape)
        return logits


class ConditionalClassPermuter(tf.keras.layers.Layer):
    """
    Aplica uma máscara de permissão de classe baseada no 'tipo de shape' latente.

    Assumimos que o modelo primeiro gera logits para classes latentes (ou embeddings),
    e depois isso é mapeado para classes reais, mas só onde for permitido.
    """
    def __init__(self, class_mask, use_logit_penalty=True):
        """
        class_mask: tensor shape [NUM_CLASSES], ou [1, 1, 1, NUM_CLASSES]
        que indica quais classes são permitidas. Pode ser aprendida ou fixa.
        """
        super().__init__()
        self.class_mask = tf.constant(class_mask, dtype=tf.float32)
        self.use_logit_penalty = use_logit_penalty

    def call(self, logits):
        """
        logits: Tensor [B, H, W, NUM_CLASSES]
        """
        if self.use_logit_penalty:
            large_neg = tf.constant(-1e9, dtype=logits.dtype)
            penalty_mask = 1.0 - self.class_mask  # zeros onde permitido
            penalty = penalty_mask * large_neg
            return logits + penalty
        else:
            return logits * self.class_mask  # simplesmente zera onde proibido
        

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
