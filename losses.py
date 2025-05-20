import tensorflow as tf

PAD_VALUE = -1  # ou qualquer valor que você tenha certeza que não está no VOCAB
# Garanta que seu padding usa isso

def masked_sparse_categorical_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, PAD_VALUE), tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-6)


def masked_loss_with_smoothing(y_true, y_pred):
    return masked_sparse_categorical_loss(y_true, y_pred)  # customize if needed
