# losses.py

import tensorflow as tf

def masked_focal_loss_wrapper(gamma=2.0, alpha=None, ignore_class=None):
    def loss_fn(y_true, y_pred):
        mask = tf.ones_like(y_true, dtype=tf.float32)
        if ignore_class is not None:
            mask = tf.cast(tf.not_equal(y_true, ignore_class), tf.float32)

        loss_obj = SparseFocalLoss(gamma=gamma, alpha=alpha)
        raw_loss = loss_obj(y_true, y_pred)
        masked_loss = raw_loss * mask
        return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-6)
    return loss_fn



class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, name="sparse_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha  # alpha pode ser dict ou lista

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_true_onehot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])

        cross_entropy = -tf.math.log(tf.reduce_sum(y_pred * y_true_onehot, axis=-1) + 1e-8)
        probs = tf.reduce_sum(y_pred * y_true_onehot, axis=-1)

        if self.alpha is not None:
            if isinstance(self.alpha, dict):
                alpha_weights = tf.constant([self.alpha.get(i, 1.0) for i in range(tf.shape(y_pred)[-1])], dtype=tf.float32)
            else:
                alpha_weights = tf.constant(self.alpha, dtype=tf.float32)
            class_alphas = tf.gather(alpha_weights, y_true)
            focal_loss = class_alphas * tf.pow(1.0 - probs, self.gamma) * cross_entropy
        else:
            focal_loss = tf.pow(1.0 - probs, self.gamma) * cross_entropy

        return tf.reduce_mean(focal_loss)

