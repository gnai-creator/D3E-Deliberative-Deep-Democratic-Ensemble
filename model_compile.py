import os
import tensorflow as tf
import tensorflow.keras as keras
from runtime_utils import log
from losses import masked_loss_with_smoothing
from core import SageAxiom

def model_compilation(index, learning_rate, vocab_size, block_index, result_dir):
    model = SageAxiom(hidden_dim=128)
    dummy_input = tf.zeros((1, 30, 30, vocab_size), dtype=tf.float32)
    _ = model(dummy_input, training=False)

    model_dir = os.path.join(result_dir, f"block_{block_index}")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{index}")

    if os.path.exists(model_path + "_weights.h5"):
        model.load_weights(model_path + "_weights.h5")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=masked_loss_with_smoothing,
        metrics=["accuracy"]
    )
    log(f"Modelo {index} compilado!")
    return model, model_path
   