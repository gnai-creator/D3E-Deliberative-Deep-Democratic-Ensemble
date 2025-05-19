# losses.py

import tensorflow as tf

class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=True, reduction=tf.keras.losses.Reduction.AUTO, name='sparse_focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        # y_true shape: (batch, height, width)
        # y_pred shape: (batch, height, width, num_classes)

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])  # shape: (B, H, W, C)

        # Clip predictions to prevent log(0) error
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        cross_entropy = -tf.math.log(y_pred)
        weights = tf.pow(1.0 - y_pred, self.gamma)

        # Focal loss component
        fl = self.alpha * weights * cross_entropy
        loss = tf.reduce_sum(y_true_one_hot * fl, axis=-1)

        return tf.reduce_mean(loss)
