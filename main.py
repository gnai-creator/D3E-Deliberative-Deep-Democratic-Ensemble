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
from shape_locator_net import ShapeLocatorNet, compile_shape_locator
import traceback

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
LEN_TRAINING = 5
FULL_TRAIN_EPOCHS = 5
PERMUTE_TRAIN_EPOCHS = 3
CYCLES = 5
MAX_BLOCKS = 400
HOURS = 12
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

def to_numpy_safe(x):
    if isinstance(x, tf.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)

for i in range(LEN_TRAINING):
    block_index = 0
    while block_index < MAX_BLOCKS and time.time() - start_time < MAX_TRAINING_TIME:
        log(f"Treinando bloco {block_index:02d}")

        X_train, X_val, Y_train, Y_val, sw_train, sw_val, info_train, info_val = get_dataset(
            block_index=block_index,
            task_ids=task_ids,
            challenges=train_challenges,
            block_size=BLOCK_SIZE,
            pad_value=PAD_VALUE,
            vocab_size=VOCAB_SIZE
        )

        print("Valores únicos de Y_train:", np.unique(Y_train.numpy()))

        if len(X_train.shape) == 4:
            X_train = X_train[..., tf.newaxis, :]
            X_val = X_val[..., tf.newaxis, :]

        shape_mask_train = tf.cast(Y_train > 0, tf.float32)[..., tf.newaxis]
        shape_mask_val = tf.cast(Y_val > 0, tf.float32)[..., tf.newaxis]

        model = ShapeLocatorNet(hidden_dim=256)
        model = compile_shape_locator(model, lr=LEARNING_RATE)

        bloco_resolvido = False

        for cycle in range(CYCLES):
            if bloco_resolvido:
                break

            log(f"Ciclo {cycle+1}/{CYCLES} - Treinando modelo ShapeLocatorNet")
            t0 = time.time()
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
            t1 = time.time()
            log(f"Treinamento do ciclo {cycle+1} concluído em {t1 - t0:.2f}s")

            try:
                x_val_sample = X_val[:1]
                y_val_sample = Y_val[:1]
                x_val_sample = transform_input(x_val_sample)

                preds = model.predict(x_val_sample)

                if isinstance(preds, dict):
                    presence_raw = preds["presence_map"]
                    presence = presence_raw[0, :, :, 0] if isinstance(presence_raw, np.ndarray) else presence_raw.numpy()[0, :, :, 0]
                else:
                    log("[WARN] Modelo retornou array, não dicionário. Usando fallback sem presence_map.")
                    presence = np.zeros_like(preds[0, :, :, 0])

                print("Presence média:", np.mean(presence))

                if np.mean(presence) < 0.01:
                    print("[INFO] Modelo acha que não tem nada aqui — talvez esteja certo?")

                y_val_logits = preds["class_logits"] if isinstance(preds, dict) else preds

                y_val_pred = tf.argmax(y_val_logits, axis=-1).numpy()
                y_val_pred_sample = y_val_pred[0]
                y_val_expected = to_numpy_safe(y_val_sample[0])

                plot_confusion(y_val_expected, y_val_pred_sample,
                               model_name=f"block_{block_index}_cycle_{cycle+1}_model_0")
                plot_prediction_debug(
                    X_val[0], y_val_expected, y_val_pred_sample,
                    model_index=f"block_{block_index}_model_0", index=cycle, pad_value=PAD_VALUE
                )

                match_pixels = (y_val_pred_sample == y_val_expected).astype(np.float32)
                presence_mask = (y_val_expected > 0).astype(np.float32)
                relevant = np.sum(presence_mask)
                color_match_pct = 1.0 if relevant == 0 else np.sum(match_pixels * presence_mask) / relevant
                shape_match_pct = np.mean(presence)

                log(f"[INFO] Color Match no bloco {block_index}, ciclo {cycle+1}: {color_match_pct*100:.2f}%")
                log(f"[INFO] Shape Match no bloco {block_index}, ciclo {cycle+1}: {shape_match_pct*100:.2f}%")

                if color_match_pct >= 1.0 and shape_match_pct >= 1.0:
                    log(f"[SUCCESS] Bloco {block_index} resolvido com sucesso. Pulando para próximo bloco.")
                    block_index += 1
                    bloco_resolvido = True

            except Exception as e:
                log(f"[ERROR] Erro ao gerar predicoes: {e}")
                traceback.print_exc()

        log(f"Predições salvas para ciclo {cycle+1} do bloco {block_index}")