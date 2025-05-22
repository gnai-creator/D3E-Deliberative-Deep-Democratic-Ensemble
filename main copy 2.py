import os
import json
import time
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from core import SageUNet
from metrics_utils import plot_history, plot_confusion, plot_prediction_debug, plot_raw_input_preview
from runtime_utils import log, profile_time, ensure_batch_dim
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sage_debate_loop import conversational_loop
from data_preparation import get_dataset
from model_compile import model_compilation
from metrics import compute_metrics
from callbacks import MetricsCallback, PermutationRegularizationCallback, EnableRefinerCallback
from training_utils import freeze_all_except_learned_color_permutation, unfreeze_all, freeze_color_permutation, unfreeze_color_permutation

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
RL_PATIENCE = 5
RL_LEARNING_RATE = 2e-3
FACTOR = 0.65
BATCH_SIZE = 8
EPOCHS = 60
RESULTS_DIR = "results"
MODELS_PER_TASK = 1
BLOCK_SIZE = 1
LEN_TRAINING = 25
FULL_TRAIN_EPOCHS = 5
PERMUTE_TRAIN_EPOCHS = 3
CYCLES = 4
MAX_BLOCKS = 1
HOURS = 1
MAX_TRAINING_TIME = HOURS * 60 * 60
MAX_EVAL_TIME = HOURS * 2 * 60 * 60 / MODELS_PER_TASK

# Load data
with open("arc-agi_training_challenges.json") as f:
    train_challenges = json.load(f)
with open("arc-agi_training_solutions.json") as f:
    train_solutions = json.load(f)
with open("arc-agi_evaluation_challenges.json") as f:
    eval_challenges = json.load(f)
with open("arc-agi_evaluation_solutions.json") as f:
    eval_solutions = json.load(f)

os.makedirs(RESULTS_DIR, exist_ok=True)
task_ids = list(train_challenges.keys())

start_time = time.time()
scores = {}
submission_dict = {}
evaluation_logs = {}

def transform_input(x):
    return tf.expand_dims(x, 0) if len(x.shape) == 4 else x

for i in range(LEN_TRAINING):
    block_index = 0
    while block_index < MAX_BLOCKS and time.time() - start_time < MAX_TRAINING_TIME:
        X_train_final, X_val_final, y_train_final, y_val_final, sw_train, sw_val, X_test_final, X_raw = get_dataset(block_index, task_ids, train_challenges, BLOCK_SIZE, PAD_VALUE, VOCAB_SIZE)
        if len(X_raw) == 0 or len(y_val_final) == 0:
            log(f"[WARN] Nenhuma amostra de validacao disponivel no bloco {block_index}")
            block_index += 1
            continue

        log(f"X_train_final shape: {X_train_final.shape}")
        plot_raw_input_preview(X_raw[0], model_name=f"block_{block_index}_task_{task_ids[block_index]}_input")

        oscillation = math.sin(block_index)
        scaling = 1 + 0.5 * oscillation
        learning_rate = LEARNING_RATE * scaling
        rl_learning_rate = RL_LEARNING_RATE * scaling
        model_path = ""
        models = []

        for i in range(MODELS_PER_TASK):
            model, model_path , base_model = model_compilation(
                index=i,
                learning_rate=learning_rate,
                vocab_size=VOCAB_SIZE,
                block_index=block_index,
                result_dir=RESULTS_DIR
            )
            models.append(model)
            x = X_train_final
            y = y_train_final
            sw = sw_train
           
            log(f"Modelo {i} vai treinar a task {task_ids[block_index]}")
            # log(f"[DEBUG] Shape x: {x.shape} | y: {y.shape} | sw: {sw.shape}")

            history_list = []
            for cycle in range(CYCLES):
                log(f"\nCiclo {cycle+1}/{CYCLES} - Treinamento completo da rede")
                unfreeze_all(model)
                freeze_color_permutation(model)
                base_model.use_refiner.assign(False)
                

                log(f"x: {x.shape}")
                log(f"y main_output: {y.shape}")
                log(f"y aux_output: {y.shape}")
                log(f"sample_weight: {sw.shape}")
                log(f"sample_weight numpy {sw.numpy()}")

                shape_mask = tf.reduce_sum(x[:, :, :, -1, :], axis=-1, keepdims=True)
                shape_mask = tf.cast(shape_mask > 0, tf.float32)
                tf.print("[DEBUG] shape_mask max/min:", tf.reduce_max(shape_mask), tf.reduce_min(shape_mask))


                history_full = model.fit(
                    x=x,
                    y={"main_output": y, "aux_output": y,"shape_output": shape_mask,},
                    validation_data=(
                        X_val_final,
                        {"main_output": y_val_final, "aux_output": y_val_final},
                        {"main_output": sw_val, "aux_output": sw_val}
                    ),
                    sample_weight={
                        "main_output": shape_mask[..., 0],
                        "aux_output": shape_mask[..., 0],
                        "shape_output": shape_mask
                    },
                    batch_size=1,
                    epochs=FULL_TRAIN_EPOCHS,
                    verbose=1,
                    callbacks=[
                        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
                        ReduceLROnPlateau(monitor="val_loss", factor=FACTOR, patience=RL_PATIENCE, min_lr=rl_learning_rate),
                        MetricsCallback(x, y_val_final, num_classes=10),
                        PermutationRegularizationCallback(),
                        EnableRefinerCallback(enable_epoch=3)
                    ]
                )
                output = model.predict(x[0:1])
                predicao = tf.argmax(output["main_output"][0], axis=-1)
                log(f"[DEBUG] PREDICAO: {predicao}")
                history_list.append(history_full)


                weights_file = model_path + "_weights"
                if os.path.exists(weights_file):
                    os.remove(weights_file)
                base_model.save_weights(weights_file, save_format='tf')
                plot_history(history_list[0], model_name=f"block_{block_index}_task_{task_ids[block_index]}_model_{i}")

               # Dentro do bloco try após treinamento do modelo
                try:
                    # Usa uma única amostra para evitar erro de cardinalidade
                    x_val_sample = X_val_final[:1]
                    y_val_sample = y_val_final[:1]
                    x_val_sample = transform_input(x_val_sample)

                    preds = model.predict(x_val_sample)
                    y_val_logits = preds.get("main_output") if isinstance(preds, dict) else preds[0] if isinstance(preds, (tuple, list)) else preds

                    if isinstance(y_val_logits, tf.Tensor) and y_val_logits.dtype == tf.string:
                        raise TypeError("[ERROR] y_val_logits do tipo string. Isso nao deveria acontecer.")
                    if isinstance(y_val_logits, np.ndarray) and y_val_logits.dtype.kind in {'U', 'S', 'O'}:
                        raise TypeError("[ERROR] y_val_logits contem strings ou objetos invalidos.")

                    y_val_pred = tf.argmax(y_val_logits, axis=-1).numpy()
                    unique_preds = np.unique(y_val_pred)
                    log(f"[DEBUG] Valores unicos preditos: {unique_preds}")

                    y_val_expected = y_val_sample.numpy()[0]
                    y_val_pred_sample = y_val_pred[0] if len(y_val_pred.shape) > 0 else y_val_pred

                    plot_confusion(y_val_expected, y_val_pred_sample, model_name=f"block_{block_index}_task_{task_ids[block_index]}_model_{i}_index_{i}")
                    plot_prediction_debug(X_raw[0], y_val_expected, y_val_pred_sample, model_index=f"block_{block_index}_task_{task_ids[block_index]}_model_{i}", index=i)

                except Exception as e:
                    log(f"[ERROR] Erro ao gerar predicoes: {e}")


                log(f"\nCiclo {cycle+1}/{CYCLES} - Treinamento da permutacao apenas")
                freeze_all_except_learned_color_permutation(model)
                unfreeze_color_permutation(model)
                base_model.use_refiner.assign(True)

                log(f"x: {x.shape}")
                log(f"y main_output: {y.shape}")
                log(f"y aux_output: {y.shape}")
                log(f"sample_weight: {sw.shape}")
                log(f"sample_weight numpy {sw.numpy()}")

                shape_mask = tf.reduce_sum(x[:, :, :, -1, :], axis=-1, keepdims=True)
                shape_mask = tf.cast(shape_mask > 0, tf.float32)
                tf.print("[DEBUG] shape_mask max/min:", tf.reduce_max(shape_mask), tf.reduce_min(shape_mask))

                history_perm = model.fit(
                    x=x,
                    y={"main_output": y, "aux_output": y,"shape_output": shape_mask,},
                    validation_data=(
                        X_val_final,
                        {"main_output": y_val_final, "aux_output": y_val_final},
                        {"main_output": sw_val, "aux_output": sw_val}
                    ),
                    sample_weight={
                        "main_output": shape_mask[..., 0],
                        "aux_output": shape_mask[..., 0],
                        "shape_output": shape_mask
                    },
                    batch_size=1,
                    epochs=PERMUTE_TRAIN_EPOCHS,
                    verbose=1,
                    callbacks=[
                        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
                        ReduceLROnPlateau(monitor="val_loss", factor=FACTOR, patience=RL_PATIENCE, min_lr=rl_learning_rate),
                        MetricsCallback(x, y_val_final, num_classes=10),
                        PermutationRegularizationCallback(),
                        EnableRefinerCallback(enable_epoch=3)
                    ]
                )

                

                history_list.append(history_perm)

                weights_file = model_path + "_weights"
                if os.path.exists(weights_file):
                    os.remove(weights_file)
                base_model.save_weights(weights_file, save_format='tf')
                plot_history(history_list[0], model_name=f"block_{block_index}_task_{task_ids[block_index]}_model_{i}")

               # Dentro do bloco try após treinamento do modelo
                try:
                    # Usa uma única amostra para evitar erro de cardinalidade
                    x_val_sample = X_val_final[:1]
                    y_val_sample = y_val_final[:1]
                    x_val_sample = transform_input(x_val_sample)

                    preds = model.predict(x_val_sample)
                    y_val_logits = preds.get("main_output") if isinstance(preds, dict) else preds[0] if isinstance(preds, (tuple, list)) else preds

                    if isinstance(y_val_logits, tf.Tensor) and y_val_logits.dtype == tf.string:
                        raise TypeError("[ERROR] y_val_logits do tipo string. Isso nao deveria acontecer.")
                    if isinstance(y_val_logits, np.ndarray) and y_val_logits.dtype.kind in {'U', 'S', 'O'}:
                        raise TypeError("[ERROR] y_val_logits contem strings ou objetos invalidos.")

                    y_val_pred = tf.argmax(y_val_logits, axis=-1).numpy()
                    unique_preds = np.unique(y_val_pred)
                    log(f"[DEBUG] Valores unicos preditos: {unique_preds}")

                    y_val_expected = y_val_sample.numpy()[0]
                    y_val_pred_sample = y_val_pred[0] if len(y_val_pred.shape) > 0 else y_val_pred

                    plot_confusion(y_val_expected, y_val_pred_sample, model_name=f"block_{block_index}_task_{task_ids[block_index]}_model_{i}_index_{i}")
                    plot_prediction_debug(X_raw[0], y_val_expected, y_val_pred_sample, model_index=f"block_{block_index}_task_{task_ids[block_index]}_model_{i}", index=i)

                except Exception as e:
                    log(f"[ERROR] Erro ao gerar predicoes: {e}")



        block_index += 1