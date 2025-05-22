import tensorflow as tf

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, penalize_fp=True, name="focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = tf.convert_to_tensor(alpha, dtype=tf.float32) if alpha is not None else None
        self.penalize_fp = penalize_fp

    def call(self, y_true, y_pred, sample_weight=None):
        probs = tf.nn.softmax(y_pred, axis=-1)
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(probs)[-1])
        probs = tf.clip_by_value(probs, 1e-9, 1.0)

        ce = -y_true_onehot * tf.math.log(probs)
        weight = tf.pow(1. - probs, self.gamma)

        # Máscara para ignorar classe 0 corretamente
        ignore_mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)
        ce *= ignore_mask
        weight *= ignore_mask

        if self.alpha is not None:
            alpha_weight = tf.reduce_sum(self.alpha * y_true_onehot, axis=-1, keepdims=True)
            weight *= alpha_weight

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            if len(sample_weight.shape) < len(ce.shape):
                sample_weight = tf.expand_dims(sample_weight, axis=-1)
            ce *= sample_weight
            weight *= sample_weight

        focal = weight * ce
        loss = tf.reduce_sum(focal, axis=-1)
        loss = tf.reduce_mean(loss)

        # Regularização da permutation
        if hasattr(self, 'model'):
            for layer in self.model.layers:
                if hasattr(layer, 'learned_color_permutation'):
                    matrix = layer.learned_color_permutation.permutation_weights
                    identity = tf.eye(tf.shape(matrix)[0])
                    matrix_no_zero = matrix[1:, :]
                    identity_no_zero = identity[1:, :]
                    reg_loss = tf.reduce_sum(tf.square(matrix_no_zero - identity_no_zero))
                    loss += 0.05 * reg_loss

        # Penalidade para falsos positivos em classe 0
        if self.penalize_fp:
            y_pred_labels = tf.argmax(probs, axis=-1, output_type=tf.int32)
            false_positives = tf.logical_and(tf.equal(y_true, 0), tf.not_equal(y_pred_labels, 0))
            penalty = tf.reduce_mean(tf.cast(false_positives, tf.float32))
            loss += 0.2 * penalty

        return loss