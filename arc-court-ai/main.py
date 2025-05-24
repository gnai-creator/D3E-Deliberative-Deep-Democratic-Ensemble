import os
import json
import time
import math
import random
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from metrics_utils import plot_prediction_debug, plot_prediction_test
from runtime_utils import log, save_debug_result, transform_input, to_numpy_safe
from data_preparation import get_dataset
from model_loader import load_model
from court_logic import arc_court


def test_challenge(models, X_test, raw_test_inputs, block_index, task_id, submission_dict):
    log(f"[TEST] Inicializando teste bloco {block_index} para task {task_id}")
    try:
        x_test_sample = tf.convert_to_tensor(X_test[0], dtype=tf.float32)
        x_test_sample = tf.expand_dims(x_test_sample, axis=0)
        
        x_test_sample = tf.convert_to_tensor(x_test_sample, dtype=tf.float32)
        if x_test_sample.ndim == 4:
            x_test_sample = tf.expand_dims(x_test_sample, axis=3)  # (B, H, W, 1, T)
        
        preds = arc_court(models, x_test_sample)

        y_test_logits = preds["class_logits"] if isinstance(preds, dict) else preds
        pred_np = tf.argmax(y_test_logits, axis=-1).numpy()[0]

        plot_prediction_test(
            raw_input=raw_test_inputs[0],
            predicted_output=pred_np,
            task_id=task_id,
            filename=f"block_{block_index}_task_{task_id}_model_0",
            index=block_index,
            pad_value=PAD_VALUE
        )
        plot_prediction_test(
            raw_input=raw_test_inputs[0],
            predicted_output=pred_np,
            task_id=task_id,
            filename=f"PREDICT TEST JUDGE RESULTS AAASSSSXXX",
            index=block_index,
            pad_value=PAD_VALUE
        )

        submission_dict.append({"task_id": task_id, "prediction": pred_np})
        save_debug_result(submission_dict, "submission.json")
    except Exception as e:
        log(f"[ERROR] Erro ao gerar predicoes: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Config
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
    CYCLES = 150
    MAX_BLOCKS = 400
    MAX_TRAINING_TIME = 4 * 60 * 60
    N_MODELS = 5

    with open("arc-court-ai/data/arc-agi_test_challenges.json") as f:
        test_challenges = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    task_ids = list(test_challenges.keys())
    start_time = time.time()
    submission_dict = []
    blacklist_blocks = [ 73, 124 ]

    for _ in range(LEN_TRAINING):
        block_index = 0
        while block_index < MAX_BLOCKS and time.time() - start_time < MAX_TRAINING_TIME:
            if block_index in blacklist_blocks:
                block_index +=1
                continue
            log(f"Treinando bloco {block_index:02d}")
            

            X_train, X_val, Y_train, Y_val, _, _, X_test, info_train, info_val, task_id, raw_inputs, raw_test_inputs = get_dataset(
                block_index=block_index, task_ids=task_ids, challenges=test_challenges,
                block_size=BLOCK_SIZE, pad_value=PAD_VALUE, vocab_size=VOCAB_SIZE
            )

            if len(X_train.shape) == 4:
                X_train = X_train[..., tf.newaxis, :]
                X_val = X_val[..., tf.newaxis, :]

            flip_targets = np.zeros((Y_train.shape[0],), dtype=np.int32)
            rotation_targets = np.zeros((Y_train.shape[0],), dtype=np.int32)
            flip_val_targets = np.zeros((Y_val.shape[0],), dtype=np.int32)
            rotation_val_targets = np.zeros((Y_val.shape[0],), dtype=np.int32)
            models = []

            for i in range(N_MODELS) :
                
                model = load_model(i, LEARNING_RATE)
                if model is None:
                    log(f"[ERROR] Modelo {i} retornou None do load_model. Pulando.")
                    continue

                if len(models) <= i:
                    models.append(model)
                else:
                    models[i] = model

                for cycle in range(CYCLES):
                    log(f"Cycle {cycle}")
                    log(f"X_train.shape: {X_train.shape}")
                    log(f"Y_train.shape: {Y_train.shape}")

                    models[i].fit(
                        x=X_train,
                        y=Y_train,
                        validation_data=(X_val, Y_val),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[
                                EarlyStopping(monitor="val_shape_acc", patience=PATIENCE, restore_best_weights=True),
                                ReduceLROnPlateau(monitor="val_loss", factor=FACTOR, patience=2, min_lr=RL_LEARNING_RATE)

                        ],
                        verbose=2
                    )

                    try:
                        x_val_sample = transform_input(X_val[:1])
                        preds = model.predict(x_val_sample)
                        y_val_logits = preds["class_logits"] if isinstance(preds, dict) else preds
                        y_val_pred = tf.argmax(y_val_logits, axis=-1).numpy()[0]
                        y_val_expected = to_numpy_safe(Y_val[:1][0])

                        valid_mask = (y_val_expected != PAD_VALUE)
                        if np.sum(valid_mask) == 0:
                            log(f"[WARN] Nenhum pixel valido para comparacao. Task: {task_id}")
                            continue

                        pixel_color_perfect = np.mean((y_val_pred == y_val_expected)[valid_mask])
                        pixel_shape_perfect = np.mean((y_val_pred > 0) == (y_val_expected > 0))

                        plot_prediction_debug(
                            raw_input=raw_inputs[0],
                            expected_output=y_val_expected,
                            predicted_output=y_val_pred,
                            model_index=f"block_{block_index}_task_{task_id}_model_0",
                            pad_value=PAD_VALUE,
                            index=block_index,
                            task_id=task_id

                        )
                        plot_prediction_debug(
                            raw_input=raw_inputs[0],
                            expected_output=y_val_expected,
                            predicted_output=y_val_pred,
                            model_index=f"AAASSSSXXX",
                            pad_value=PAD_VALUE,
                            index=block_index,
                            task_id=task_id
                        )

                        log(f"Pixel Color Perfect: {pixel_color_perfect} - Pixel Shape Perfect {pixel_shape_perfect}")
                        if pixel_color_perfect >= 0.999 and pixel_shape_perfect >= 0.999:
                            test_challenge(models, X_test, raw_test_inputs, block_index, task_id, submission_dict)
                            block_index += 1
                            break

                    except Exception as e:
                        log(f"[ERROR] Erro ao avaliar bloco {block_index}: {e}")
                        traceback.print_exc()
