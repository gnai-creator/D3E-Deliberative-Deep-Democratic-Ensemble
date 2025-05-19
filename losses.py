# losses.py

import tensorflow as tf

VOCAB_SIZE = 10
    

def dynamic_focal_loss_wrapper(alpha_var, gamma=2.0):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)                         # [B, H, W]
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)               # [B, H, W, C]
        
        # One-hot com broadcast implícito
        y_true_onehot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])  # [B, H, W, C]

        # Pega prob da classe correta
        p_t = tf.reduce_sum(y_pred * y_true_onehot, axis=-1)       # [B, H, W]
        alpha_t = tf.gather(alpha_var, y_true)                     # [B, H, W]

        focal_weight = tf.pow(1.0 - p_t, gamma)
        loss = -alpha_t * focal_weight * tf.math.log(tf.clip_by_value(p_t, 1e-5, 1.0))

        return tf.reduce_mean(loss)
    
    return focal_loss


class AlphaWarmupCallback(tf.keras.callbacks.Callback):
    def __init__(self, alpha_var, initial_alpha, target_alpha, warmup_epochs):
        super().__init__()
        self.alpha_var = alpha_var
        self.initial_alpha = initial_alpha
        self.target_alpha = target_alpha
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        progress = min(epoch / self.warmup_epochs, 1.0)
        new_alpha = self.initial_alpha + (self.target_alpha - self.initial_alpha) * progress
        self.alpha_var.assign(new_alpha)

