import os
import tensorflow as tf
import tensorflow.keras as keras
from runtime_utils import log
from losses import FocalLoss
from core import SageUNet

def model_compilation(index, learning_rate, vocab_size, block_index, result_dir):
    base_model = SageUNet(hidden_dim=128)
    
    inputs = tf.keras.Input(shape=(30, 30, 10, vocab_size))
    final_logits, base_logits = base_model(inputs)
    model = tf.keras.Model(
        inputs=inputs,
        outputs={"main_output": final_logits, "aux_output": base_logits}
    )

    model_dir = os.path.join(result_dir, f"Model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{index}")

    if os.path.exists(model_path + "_weights.h5"):
        model.load_weights(model_path + "_weights.h5")

    focal_loss = FocalLoss(
        gamma=1.0,
        alpha=[0.3] + [1.33] * 9,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={  "main_output":  focal_loss,
                "aux_output":  focal_loss
        },
        loss_weights={"main_output": 0.8, "aux_output": 0.05},
        metrics={"main_output": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
             "aux_output":  [tf.keras.metrics.SparseCategoricalAccuracy(name="acc_aux")]}
    )


    log(f"Modelo {index} compilado!")
    return model, model_path
