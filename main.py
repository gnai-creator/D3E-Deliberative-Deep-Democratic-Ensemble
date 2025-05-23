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

from metrics import pad_to_shape_batch, match_grid_orientation
from metrics_utils import plot_prediction_debug, plot_prediction_test
from runtime_utils import log, save_debug_result, transform_input, to_numpy_safe
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preparation import get_dataset
from core import SimuV1
from model_compile import compile_model

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

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

with open("arc-agi_test_challenges.json") as f:
    test_challenges = json.load(f)

os.makedirs(RESULTS_DIR, exist_ok=True)
task_ids = list(test_challenges.keys())
start_time = time.time()
submission_dict = []


def test_challenge(model, X_test,raw_test_inputs, block_index, task_id, submission_dict, EVAL_CYCLES=20, PAD_VALUE=0):
    log(f"[TEST] Inicializando teste bloco {block_index} para task {task_id}")
    try:
        x_test_sample = tf.convert_to_tensor(X_test[0], dtype=tf.float32)
        x_test_sample = tf.expand_dims(x_test_sample, axis=0)
        preds = model(x_test_sample, training=False)

        if isinstance(preds, dict):
            y_test_logits = preds["class_logits"]
        else:
            y_test_logits = preds

        y_test_logits = match_grid_orientation(X_test[0], y_test_logits)
        pred_np = tf.argmax(y_test_logits, axis=-1).numpy()[0]

        plot_prediction_test(
            raw_input=raw_test_inputs[0], 
            predicted_output=pred_np, task_id=task_id,
            filename=f"block_{block_index}_task_{task_id}_model_0",
            index=0, pad_value=PAD_VALUE
        )

        submission_dict.append({"task_id": task_id, "prediction": pred_np})
        save_debug_result(submission_dict, "submission.json")

    except Exception as e:
        log(f"[ERROR] Erro ao gerar predicoes: {e}")
        traceback.print_exc()

for _ in range(LEN_TRAINING):
    block_index = 0
    while block_index < MAX_BLOCKS and time.time() - start_time < MAX_TRAINING_TIME:
        log(f"Treinando bloco {block_index:02d}")

        model = SimuV1(hidden_dim=512)
        model = compile_model(model, lr=LEARNING_RATE)

        X_train, X_val, Y_train, Y_val, _, _, X_test, info_train, info_val, task_id, raw_inputs, raw_test_inputs = get_dataset(
            block_index=block_index, task_ids=task_ids, challenges=test_challenges,
            block_size=BLOCK_SIZE, pad_value=PAD_VALUE, vocab_size=VOCAB_SIZE
        )

        if len(X_train.shape) == 4:
            X_train = X_train[..., tf.newaxis, :]
            X_val = X_val[..., tf.newaxis, :]

        bloco_resolvido = False

        for cycle in range(CYCLES):
            if bloco_resolvido:
                break

            log(f"X_train.shape {X_train.shape}")
            log(f"Y_train.shape {Y_train.shape}")

            pred_shape = model(X_train[:1], training=False)
            pred_grid = pred_shape["class_logits"] if isinstance(pred_shape, dict) else pred_shape
            pred_grid = match_grid_orientation(Y_train[:1], pred_grid)
            pred_shape_hw = pred_grid.shape[1:3]

            log(f"Model Output Shape: {pred_grid.shape}")
            log(f"Y_train Shape: {Y_train.shape}")

            if Y_train.ndim == 2:
                log(f"[WARN] Y_train shape was {Y_train.shape}, reshaping...")
                Y_train = Y_train[np.newaxis, :, :]

            if Y_train.shape[1:3] != pred_shape_hw:
                Y_train = pad_to_shape_batch(Y_train, pred_shape_hw, pad_value=PAD_VALUE)
                Y_val = pad_to_shape_batch(Y_val, pred_shape_hw, pad_value=PAD_VALUE)

            log(f"Before training: Y_train.shape = {Y_train.shape}, Y_train dtype = {Y_train.dtype}")
            if Y_train.ndim == 4 and Y_train.shape[-1] == 1:
                log("[WARN] Removendo eixo extra de Y_train")
                Y_train = np.squeeze(Y_train, axis=-1)

            if Y_val.ndim == 4 and Y_val.shape[-1] == 1:
                log("[WARN] Removendo eixo extra de Y_val")
                Y_val = np.squeeze(Y_val, axis=-1)

            # Cria rótulos auxiliares fixos (classe 0 para todos — sem transformação)
            flip_targets = np.zeros((Y_train.shape[0],), dtype=np.int32)
            rotation_targets = np.zeros((Y_train.shape[0],), dtype=np.int32)
            flip_val_targets = np.zeros((Y_val.shape[0],), dtype=np.int32)
            rotation_val_targets = np.zeros((Y_val.shape[0],), dtype=np.int32)

            # Treinamento com múltiplas saídas
            model.fit(
                x=X_train,
                y={
                    "class_logits": Y_train,
                    "flip_logits": flip_targets,
                    "rotation_logits": rotation_targets
                },
                validation_data=(
                    X_val,
                    {
                        "class_logits": Y_val,
                        "flip_logits": flip_val_targets,
                        "rotation_logits": rotation_val_targets
                    }
                ),
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=[
                    EarlyStopping(monitor="val_class_logits_loss", patience=PATIENCE, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_class_logits_loss", factor=FACTOR, patience=2, min_lr=RL_LEARNING_RATE)
                ]
            )


            try:
                x_val_sample = transform_input(X_val[:1])
                preds = model.predict(x_val_sample)
                y_val_logits = preds["class_logits"] if isinstance(preds, dict) else preds
                y_val_logits = match_grid_orientation(Y_val[:1], y_val_logits)
                y_val_pred = tf.argmax(y_val_logits, axis=-1).numpy()[0]
                y_val_expected = to_numpy_safe(Y_val[:1][0])

                valid_mask = (y_val_expected != PAD_VALUE)
                if np.sum(valid_mask) == 0:
                    log(f"[WARN] Nenhum pixel valido para comparacao. Task: {task_id}")
                    continue

                pixel_color_perfect = np.mean((y_val_pred == y_val_expected)[valid_mask])
                pixel_shape_perfect = np.mean((y_val_pred > 0) == (y_val_expected > 0))

                plot_prediction_debug(
                    raw_input=raw_inputs[0], expected_output=y_val_expected, predicted_output=y_val_pred,
                    model_index=f"block_{block_index}_task_{task_id}_model_0",
                    index=cycle, pad_value=PAD_VALUE
                )

                log(f"Pixel Color Perfect: {pixel_color_perfect} - Pixel Shape Perfect {pixel_shape_perfect}")

                if pixel_color_perfect >= 0.999 and pixel_shape_perfect >= 0.999:
                    bloco_resolvido = True
                    test_challenge(model, X_test, raw_test_inputs, block_index, task_id, submission_dict)
                    block_index += 1
                    break

            except Exception as e:
                log(f"[ERROR] Erro ao avaliar bloco {block_index}: {e}")
                traceback.print_exc()