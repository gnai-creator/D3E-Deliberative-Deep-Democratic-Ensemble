import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Remover logs do TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

def masked_loss(pad_value=-1.0, blur_penalty_weight=0.3, color_penalty_weight=0.07, ssim_weight=0.7, com_weight=0.2, valores_validos=None):
    if valores_validos is None:
        valores_validos = tf.constant([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        while y_true.shape.rank < 5:
            y_true = tf.expand_dims(y_true, axis=-1)
        while y_pred.shape.rank < 5:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        mask = tf.not_equal(y_true[:, :, :, 0:1, :], pad_value)
        mask = tf.broadcast_to(mask, tf.shape(y_true))
        mask = tf.cast(mask, tf.float32)
        y_true = tf.broadcast_to(y_true, tf.shape(y_pred))

        # MSE mascarado
        mse_loss = tf.square(y_true - y_pred) * mask
        mse = tf.reduce_sum(mse_loss) / tf.reduce_sum(mask)

        # Penalidade por borramento
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

        # SSIM
        y_true_img = tf.squeeze(y_true, axis=-1)
        y_pred_img = tf.squeeze(y_pred, axis=-1)
        if y_true_img.shape.rank == 3:
            y_true_img = tf.expand_dims(y_true_img, axis=-1)
        if y_pred_img.shape.rank == 3:
            y_pred_img = tf.expand_dims(y_pred_img, axis=-1)
        shape_check = tf.shape(y_true_img)
        h_check = shape_check[1]
        w_check = shape_check[2]
        ssim_applicable = tf.logical_and(h_check >= 11, w_check >= 11)

        def compute_ssim():
            return 1.0 - tf.reduce_mean(tf.image.ssim(y_true_img, y_pred_img, max_val=9.0))

        ssim_loss = tf.cond(ssim_applicable, compute_ssim, lambda: tf.constant(0.0))

        # Penalidade por centro de massa
        def center_of_mass(grid):
            grid = grid[:, :, :, 0, 0]  # (B, H, W)
            grid_sum = tf.reduce_sum(grid, axis=[1, 2], keepdims=True) + 1e-6
            x_coords = tf.cast(tf.range(tf.shape(grid)[2]), tf.float32)
            y_coords = tf.cast(tf.range(tf.shape(grid)[1]), tf.float32)
            x_center = tf.reduce_sum(grid * x_coords[tf.newaxis, tf.newaxis, :], axis=[1, 2]) / tf.squeeze(grid_sum)
            y_center = tf.reduce_sum(grid * y_coords[tf.newaxis, :, tf.newaxis], axis=[1, 2]) / tf.squeeze(grid_sum)
            return tf.stack([x_center, y_center], axis=1)

        com_true = center_of_mass(y_true)
        com_pred = center_of_mass(y_pred)
        com_loss = tf.reduce_mean(tf.norm(com_true - com_pred, axis=1))

        # Perda total
        total_loss = mse
        total_loss += blur_penalty_weight * blur_penalty
        total_loss += color_penalty_weight * color_penalty
        total_loss += ssim_weight * ssim_loss
        total_loss += com_weight * com_loss

        return total_loss

    return loss_fn

def compile_model(model, lr=0.001):
    optimizer = Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        # loss=masked_loss(
        #     blur_penalty_weight=0.3,
        #     pad_value=-1,
        #     color_penalty_weight=0.07,
        #     ssim_weight=0.7,
        #     com_weight=0.2
        # ),
        metrics=[tf.keras.metrics.MeanSquaredError()]
    )
    return model
