import tensorflow as tf
import tensorflow.keras as keras

PAD_VALUE = -1  # ou 255, ou qualquer coisa que você *não* use como classe real

def masked_loss_with_smoothing(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, PAD_VALUE), tf.float32)

    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    loss = tf.cast(loss, tf.float32)

    masked_loss = tf.multiply(loss, mask)
    numerator = tf.reduce_sum(masked_loss)
    denominator = tf.reduce_sum(mask) + tf.constant(1e-6, dtype=numerator.dtype)

    return numerator / denominator

def focal_loss_with_aux(gamma=2.0, alpha=None, main_weight=1.0, aux_weight=0.5):
    def compute_focal_loss(y_true, logits):
        probs = tf.nn.softmax(logits, axis=-1)
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(probs)[-1])
        probs = tf.clip_by_value(probs, 1e-9, 1.0)

        ce = -y_true_onehot * tf.math.log(probs)
        weight = tf.pow(1. - probs, gamma)

        if alpha is not None:
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            alpha_weight = tf.reduce_sum(alpha_tensor * y_true_onehot, axis=-1, keepdims=True)
            weight *= alpha_weight

        focal = weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))

    def loss_fn(y_true, y_pred):
        # Espera que y_true e y_pred sejam dicionários com as mesmas chaves
        main_loss = compute_focal_loss(y_true["main_output"], y_pred["main_output"])
        aux_loss = compute_focal_loss(y_true["aux_output"], y_pred["aux_output"])
        return main_weight * main_loss + aux_weight * aux_loss

    return loss_fn



def focal_loss(gamma=2.0, alpha=None):
    def loss_fn(y_true, y_pred):
        probs = tf.nn.softmax(y_pred, axis=-1)
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(probs)[-1])
        probs = tf.clip_by_value(probs, 1e-9, 1.0)

        ce = -y_true_onehot * tf.math.log(probs)
        weight = tf.pow(1. - probs, gamma)

        if alpha is not None:
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            alpha_weight = tf.reduce_sum(alpha_tensor * y_true_onehot, axis=-1, keepdims=True)
            weight *= alpha_weight

        focal = weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))

    return loss_fn








