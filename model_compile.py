import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Remover logs do TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

def masked_loss(pad_value=-1.0, blur_penalty_weight=0.1, color_penalty_weight=0.1, valores_validos=None):
    if valores_validos is None:
        valores_validos = tf.constant([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # tf.debugging.check_numerics(y_pred, "y_pred tem NaN ou Inf")
        # tf.debugging.check_numerics(y_true, "y_true tem NaN ou Inf")

        while y_true.shape.rank < 5:
            y_true = tf.expand_dims(y_true, axis=-1)
        while y_pred.shape.rank < 5:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        mask = tf.not_equal(y_true, pad_value)
        mask = tf.cast(mask, tf.float32)
        y_true = tf.broadcast_to(y_true, tf.shape(y_pred))
        mask = tf.broadcast_to(mask, tf.shape(y_pred))

        mse_loss = tf.square(y_true - y_pred) * mask
        mse = tf.reduce_sum(mse_loss) / tf.reduce_sum(mask)

        kernel_2d = tf.constant([[[0.], [1.], [0.]], [[1.], [-4.], [1.]], [[0.], [1.], [0.]]], dtype=tf.float32)
        kernel_2d = tf.reshape(kernel_2d, [3, 3, 1, 1])
        shape = tf.shape(y_pred)
        b, h, w = shape[0], shape[1], shape[2]
        y_flat = tf.reshape(y_pred, [-1, h, w, 1])
        conv = tf.nn.conv2d(y_flat, kernel_2d, strides=[1, 1, 1, 1], padding="SAME")
        blur_penalty = tf.reduce_mean(tf.square(conv))

        if y_pred.shape.rank == 5:
            y_color = tf.reshape(y_pred, [tf.shape(y_pred)[0], tf.shape(y_pred)[1], tf.shape(y_pred)[2]])
        else:
            y_color = y_pred

        valores = tf.reshape(valores_validos, [1, 1, 1, -1])
        y_color_exp = tf.expand_dims(y_color, axis=-1)
        distances = tf.abs(y_color_exp - valores)
        min_dists = tf.reduce_min(distances, axis=-1)
        color_penalty = tf.reduce_mean(min_dists)

        total_loss = mse + blur_penalty_weight * blur_penalty + color_penalty_weight * color_penalty
        return total_loss

    return loss_fn

def compile_model(model, lr=0.001):
    optimizer = Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(
                optimizer=optimizer, 
                loss=masked_loss(
                    blur_penalty_weight=0.23,
                    pad_value=-1,
                    color_penalty_weight=0.33
                ), 
                metrics=[tf.keras.metrics.MeanSquaredError()]
    )
    return model
