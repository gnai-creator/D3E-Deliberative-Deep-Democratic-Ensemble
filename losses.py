import tensorflow as tf
import tensorflow.keras as keras

PAD_VALUE = -1  # ou qualquer valor que você tenha certeza que não está no VOCAB
# Garanta que seu padding usa isso
 # customize if needed

def masked_loss_with_smoothing(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, PAD_VALUE), tf.float32)

    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    loss = tf.cast(tf.convert_to_tensor(loss), tf.float32)

    # Usa tf.math.multiply para garantir a compatibilidade
    masked_loss = tf.multiply(loss, mask)

    # make sure both are tensors
    numerator = tf.reduce_sum(masked_loss)
    denominator = tf.reduce_sum(mask) + tf.constant(1e-6, dtype=numerator.dtype)

    return numerator / denominator
