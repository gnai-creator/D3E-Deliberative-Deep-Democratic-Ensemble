import os
import tensorflow as tf
from runtime_utils import log
from tensorflow.keras.metrics import SparseCategoricalAccuracy


def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={
            "class_logits": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "flip_logits": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "rotation_logits": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        },
        loss_weights={
            "class_logits": 1.0,
            "flip_logits": 0.1,
            "rotation_logits": 0.1
        },
        metrics={
            "class_logits": [SparseCategoricalAccuracy(name="shape_acc")],
            "flip_logits": [SparseCategoricalAccuracy()],
            "rotation_logits": [SparseCategoricalAccuracy()]
        }
    )
    return model
