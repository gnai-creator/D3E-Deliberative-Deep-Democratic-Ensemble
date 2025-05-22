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
CYCLES = 1
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
        log(f"Treinando bloco {block_index:02d}")

        X_train, X_val, Y_train, Y_val, sw_train, sw_val, info_train, info_val = get_dataset(
                    block_index=block_index,
                    task_ids=task_ids,
                    challenges=train_challenges,
                    block_size=BLOCK_SIZE,
                    pad_value=PAD_VALUE,
                    vocab_size=VOCAB_SIZE
                )
        
        # log(f"X_train: {X_train}")
        # log(f"X_val: {X_val}")
        # log(f"Y_train: {Y_train}")
        # log(f"Y_val: {Y_val}")
        # log(f"Y_train: {Y_train}")



        if len(X_train.shape) == 4:
            X_train = X_train[..., tf.newaxis, :]  # adiciona T=1
            X_val = X_val[..., tf.newaxis, :]

        shape_mask_train = tf.cast(Y_train > 0, tf.float32)[..., tf.newaxis]
        shape_mask_val = tf.cast(Y_val > 0, tf.float32)[..., tf.newaxis]

        model = ShapeLocatorNet(hidden_dim=256)
        model = compile_shape_locator(model, lr=LEARNING_RATE)

        for cycle in range(CYCLES):
            log(f"Ciclo {cycle+1}/{CYCLES} - Treinando modelo ShapeLocatorNet")

            t0 = time.time()
            model.fit(
                x=X_train,
                y=shape_mask_train,
                validation_data=(X_val, shape_mask_val),
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=FACTOR, patience=2, min_lr=RL_LEARNING_RATE)
                ]
            )
            t1 = time.time()
            log(f"Treinamento do ciclo {cycle+1} concluído em {t1 - t0:.2f}s")

            preds = model.predict(X_val)

            try:
                x_val_sample = X_val[:1]
                y_val_sample = Y_val[:1]
                x_val_sample = transform_input(x_val_sample)

                preds = model.predict(x_val_sample)

                # Se for dicionário, pega a chave correta
                if isinstance(preds, dict):
                    y_val_logits = preds["class_logits"]
                else:
                    y_val_logits = preds

                # Verificação de tipo
                if isinstance(y_val_logits, tf.Tensor) and y_val_logits.dtype == tf.string:
                    raise TypeError("[ERROR] y_val_logits do tipo string. Isso nao deveria acontecer.")
                if isinstance(y_val_logits, np.ndarray) and y_val_logits.dtype.kind in {'U', 'S', 'O'}:
                    raise TypeError("[ERROR] y_val_logits contem strings ou objetos invalidos.")

                # Predição: pega argmax nos logits
                y_val_pred = tf.argmax(y_val_logits, axis=-1).numpy()
                y_val_pred_sample = y_val_pred[0]  # (H, W)
                y_val_expected = y_val_sample.numpy()[0]  # (H, W)

                if np.sum(y_val_expected) == 0 and np.sum(y_val_pred_sample) == 0:
                    log("[WARN] Nenhum pixel positivo nas predições ou ground truth — ignorando visualização.")
                else:
                    plot_confusion(y_val_expected, y_val_pred_sample,
                                model_name=f"block_{block_index}_cycle_{cycle+1}_model_0")
                    plot_prediction_debug(
                        X_val[0], y_val_expected, y_val_pred_sample,
                        model_index=f"block_{block_index}_model_0", index=cycle, pad_value=PAD_VALUE
                    )

            except Exception as e:
                log(f"[ERROR] Erro ao gerar predicoes: {e}")







            log(f"Predições salvas para ciclo {cycle+1} do bloco {block_index}")

    block_index += 1