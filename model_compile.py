import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Remover logs do TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

def masked_loss(pad_value=-1.0, blur_penalty_weight=0.23, color_penalty_weight=0.33, ssim_weight=0.1, valores_validos=None):
    if valores_validos is None:
        valores_validos = tf.constant([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Garantir que ambos têm 5 dimensões: (batch, height, width, depth, channels)
        while y_true.shape.rank < 5:
            y_true = tf.expand_dims(y_true, axis=-1)
        while y_pred.shape.rank < 5:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        # Criar máscara para ignorar valores com pad_value
        mask = tf.not_equal(y_true, pad_value)
        mask = tf.cast(mask, tf.float32)
        y_true = tf.broadcast_to(y_true, tf.shape(y_pred))
        mask = tf.broadcast_to(mask, tf.shape(y_pred))

        # MSE mascarado
        mse_loss = tf.square(y_true - y_pred) * mask
        mse = tf.reduce_sum(mse_loss) / tf.reduce_sum(mask)

        # Penalidade por borramento usando kernel Laplaciano
        kernel_2d = tf.constant([[[0.], [1.], [0.]], [[1.], [-4.], [1.]], [[0.], [1.], [0.]]], dtype=tf.float32)
        kernel_2d = tf.reshape(kernel_2d, [3, 3, 1, 1])
        shape = tf.shape(y_pred)
        b, h, w = shape[0], shape[1], shape[2]
        y_flat = tf.reshape(y_pred, [-1, h, w, 1])
        conv = tf.nn.conv2d(y_flat, kernel_2d, strides=[1, 1, 1, 1], padding="SAME")
        blur_penalty = tf.reduce_mean(tf.square(conv))

        # Penalidade por distância de cor
        y_color = tf.squeeze(y_pred, axis=-1)
        valores = tf.reshape(valores_validos, [1, 1, 1, -1])
        y_color_exp = tf.expand_dims(y_color, axis=-1)
        distances = tf.abs(y_color_exp - valores)
        min_dists = tf.reduce_min(distances, axis=-1)
        color_penalty = tf.reduce_mean(min_dists)

        # Penalidade estrutural SSIM (ajuda com forma)
        y_true_img = tf.squeeze(y_true, axis=-1)
        y_pred_img = tf.squeeze(y_pred, axis=-1)

        # Assegura que estão com 4 dimensões para tf.image.ssim
        if y_true_img.shape.rank == 3:
            y_true_img = tf.expand_dims(y_true_img, axis=-1)
        if y_pred_img.shape.rank == 3:
            y_pred_img = tf.expand_dims(y_pred_img, axis=-1)

        # Apenas aplica SSIM se o tamanho for válido (min 11x11)
        shape_check = tf.shape(y_true_img)
        h_check = shape_check[1]
        w_check = shape_check[2]
        ssim_applicable = tf.logical_and(h_check >= 11, w_check >= 11)

        def compute_ssim():
            return 1.0 - tf.reduce_mean(tf.image.ssim(y_true_img, y_pred_img, max_val=9.0))

        ssim_loss = tf.cond(ssim_applicable, compute_ssim, lambda: tf.constant(0.0))

        # Perda total
        total_loss = mse
        total_loss += blur_penalty_weight * blur_penalty
        total_loss += color_penalty_weight * color_penalty
        total_loss += ssim_weight * ssim_loss

        return total_loss

    return loss_fn

def compile_model(model, lr=0.001):
    optimizer = Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=masked_loss(
            blur_penalty_weight=0.23,
            pad_value=-1,
            color_penalty_weight=0.33,
            ssim_weight=0.1
        ),
        metrics=[tf.keras.metrics.MeanSquaredError()]
    )
    return model
