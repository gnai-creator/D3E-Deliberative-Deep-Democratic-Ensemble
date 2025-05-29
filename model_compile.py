import os
import tensorflow as tf
from runtime_utils import log
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow.keras import mixed_precision



def compile_model(model, lr=1e-3):
    base_optimizer = Adam(learning_rate=lr)
    optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)

    model.compile(
        optimizer=optimizer,
        loss = masked_mse_loss(
            pad_value=-1.0,
            blur_penalty_weight=0.15,
            color_penalty_weight=0.25,
            valores_validos=tf.constant([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        ),

        metrics=[tf.keras.metrics.MeanSquaredError()]
    )
    return model


def masked_mse_loss(pad_value=-1.0, blur_penalty_weight=0.1, color_penalty_weight=0.1, valores_validos=None):
    if valores_validos is None:
        valores_validos = tf.constant([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        while y_true.shape.rank < 5:
            y_true = tf.expand_dims(y_true, axis=-1)
        while y_pred.shape.rank < 5:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        mask = tf.not_equal(y_true, pad_value)
        mask = tf.cast(mask, tf.float32)
        y_true = tf.broadcast_to(y_true, tf.shape(y_pred))
        mask = tf.broadcast_to(mask, tf.shape(y_pred))

        # === MSE mascarado ===
        mse_loss = tf.square(y_true - y_pred) * mask
        mse = tf.reduce_sum(mse_loss) / tf.reduce_sum(mask)

        # === Penalização de borrão vetorizada ===
        kernel_2d = tf.constant([
            [[0.], [1.], [0.]],
            [[1.], [-4.], [1.]],
            [[0.], [1.], [0.]]
        ], dtype=tf.float32)
        kernel_2d = tf.reshape(kernel_2d, [3, 3, 1, 1])

        y_perm = tf.transpose(y_pred, [3, 0, 1, 2, 4])

        def penaliza_blur(slice_4d):
            conv = tf.nn.conv2d(slice_4d, kernel_2d, strides=[1, 1, 1, 1], padding="SAME")
            return tf.reduce_mean(tf.square(conv))

        blur_penalties = tf.map_fn(penaliza_blur, y_perm, fn_output_signature=tf.float32)
        blur_penalty = tf.reduce_mean(blur_penalties)

        # === Penalização por cor inválida — versão vetorizada ===
        y_flat = tf.reshape(y_pred, [-1, 1])
        distances = tf.abs(y_flat - valores_validos)
        min_dists = tf.reduce_min(distances, axis=1)
        color_penalty = tf.reduce_mean(min_dists)

        # === Loss final ===
        total_loss = mse + blur_penalty_weight * blur_penalty + color_penalty_weight * color_penalty
        return total_loss

    return loss_fn







