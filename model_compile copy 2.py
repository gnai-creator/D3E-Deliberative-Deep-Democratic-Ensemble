import os
import tensorflow as tf
import tensorflow.keras as keras
from runtime_utils import log
from losses import FocalLoss
from core import SageUNet

def model_compilation(index, learning_rate, vocab_size, block_index, result_dir):
    base_model = SageUNet(hidden_dim=256)

    inputs = tf.keras.Input(shape=(30, 30, 10, vocab_size))
    outputs = base_model(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model_dir = os.path.join(result_dir, f"Model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{index}")
    weights_file = model_path + "_weights.keras"

    if os.path.exists(weights_file):
        try:
            model.load_weights(weights_file)
        except Exception as e:
            log(f"[WARN] Falha ao carregar pesos de {weights_file}: {e}")

    focal_loss = FocalLoss(
        gamma=1.0,
        alpha=[0.1] + [1.0] * 9,
    )
    focal_loss.model = model

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "main_output": focal_loss,
            "aux_output": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "shape_output": tf.keras.losses.BinaryCrossentropy()
        },
        loss_weights={
            "main_output": 0.8,
            "aux_output": 0.2,
        },
        metrics={
            "main_output": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
            "aux_output": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc_aux")],
            "shape_output": [tf.keras.metrics.BinaryAccuracy(name="shape_acc")]
        }
    )

    log(f"Modelo {index} compilado!")
    return model, model_path, base_model
