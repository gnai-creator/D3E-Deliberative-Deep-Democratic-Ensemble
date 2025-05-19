# losses.py

import tensorflow as tf

class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2., alpha=0.25, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.nn.softmax(y_pred, axis=-1) if self.from_logits else y_pred

        y_true_one_hot = tf.one_hot(y_true, depth=y_pred.shape[-1])
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred + 1e-8)
        loss = self.alpha * tf.math.pow(1 - y_pred, self.gamma) * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

