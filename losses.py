import tensorflow as tf

def masked_sparse_categorical_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-6)
