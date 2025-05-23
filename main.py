import os
import json
import time
import math
import random
import traceback
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from metrics_utils import plot_prediction_debug, plot_prediction_test
from runtime_utils import log
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preparation import get_dataset
from shape_locator_net import ShapeLocatorNet, compile_shape_locator

# Seed and env setup
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# Hyperparameters
PAD_VALUE = 0
VOCAB_SIZE = 10
LEARNING_RATE = 0.0005
PATIENCE = 20
RL_LEARNING_RATE = 2e-3
FACTOR = 0.65
BATCH_SIZE = 8
EPOCHS = 60
RESULTS_DIR = "results"
BLOCK_SIZE = 1
LEN_TRAINING = 1
CYCLES = 20
EVAL_CYCLES = 20
MAX_BLOCKS = 400
HOURS = 4
MAX_TRAINING_TIME = HOURS * 60 * 60
MAX_EVAL_TIME = HOURS * 2 * 60 * 60

# Load test data
with open("arc-agi_test_challenges.json") as f:
    test_challenges = json.load(f)

os.makedirs(RESULTS_DIR, exist_ok=True)
task_ids = list(test_challenges.keys())
start_time = time.time()
submission_dict = []

def transform_input(x):
    return tf.expand_dims(x, 0) if len(x.shape) == 4 else x

def to_numpy_safe(x):
    return x.numpy() if isinstance(x, tf.Tensor) else np.array(x)

def test_challenge(model, X_test, block_index, task_id):
    log(f"[TEST] Inicializando teste bloco {block_index} para task {task_id}")
    for cycle in range(EVAL_CYCLES):
        try:
            x_test_sample = X_test[:1]
            preds = model.predict(x_test_sample)
            pred_np = tf.argmax(preds["class_logits"], axis=-1).numpy().tolist()
            plot_prediction_test(
                X_test[0], pred_np[0],
                filename=f"block_{block_index}_task_{task_id}_model_0",
                index=cycle, pad_value=PAD_VALUE
            )
            submission_dict.append({"task_id": task_id, "prediction": pred_np})
            with open("submission.json", "w") as f:
                json.dump(submission_dict, f)
        except Exception as e:
            log(f"[ERROR] Erro ao gerar predicoes: {e}")
            traceback.print_exc()

for _ in range(LEN_TRAINING):
    block_index = 0
    while block_index < MAX_BLOCKS and time.time() - start_time < MAX_TRAINING_TIME:
        log(f"Treinando bloco {block_index:02d}")
        model = ShapeLocatorNet(hidden_dim=256)
        model = compile_shape_locator(model, lr=LEARNING_RATE)

        X_train, X_val, Y_train, Y_val, _, _, X_test, info_train, info_val, task_id = get_dataset(
            block_index=block_index,
            task_ids=task_ids,
            challenges=test_challenges,
            block_size=BLOCK_SIZE,
            pad_value=PAD_VALUE,
            vocab_size=VOCAB_SIZE
        )

        if len(X_train.shape) == 4:
            X_train = X_train[..., tf.newaxis, :]
            X_val = X_val[..., tf.newaxis, :]

        bloco_resolvido = False

        for cycle in range(CYCLES):
            if bloco_resolvido:
                break
            model.fit(
                x=X_train,
                y=Y_train,
                validation_data=(X_val, Y_val),
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=FACTOR, patience=2, min_lr=RL_LEARNING_RATE)
                ]
            )

            try:
                x_val_sample = transform_input(X_val[:1])
                preds = model.predict(x_val_sample)
                y_val_logits = preds["class_logits"]
                y_val_pred = tf.argmax(y_val_logits, axis=-1).numpy()[0]
                y_val_expected = to_numpy_safe(Y_val[:1][0])

                valid_mask = (y_val_expected != PAD_VALUE)
                if np.sum(valid_mask) == 0:
                    log(f"[WARN] Nenhum pixel válido para comparação. Task: {task_id}")
                    continue

                pixel_color_perfect = np.mean((y_val_pred == y_val_expected)[valid_mask])
                pixel_shape_perfect = np.mean((y_val_pred > 0) == (y_val_expected > 0))

                plot_prediction_debug(
                    X_val[0], y_val_expected, y_val_pred,
                    model_index=f"block_{block_index}_task_{task_id}_model_0",
                    index=cycle, pad_value=PAD_VALUE
                )

                log(f"Pixel Color Perfect: {pixel_color_perfect} - Pixel Shape Perfect {pixel_shape_perfect}")

                if pixel_color_perfect >= 0.9 and pixel_shape_perfect >= 0.9:
                    bloco_resolvido = True
                    test_challenge(model, X_test, block_index, task_id)
                    block_index += 1
                    break

            except Exception as e:
                log(f"[ERROR] Erro ao avaliar bloco {block_index}: {e}")
                traceback.print_exc()
