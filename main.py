# main.py adaptado para compatibilidade total com vocab_size = 15

import os
import json
import time
import logging
import tensorflow as tf
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from core import SageAxiom
from metrics_utils import plot_history, plot_confusion, plot_attempts_stats
from sage_dabate_loop import conversational_loop
from runtime_utils import log, pad_to_shape, profile_time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from losses import SparseFocalLoss

# Hiperparâmetros
VOCAB_SIZE = 15
NUMBER_OF_MODELS = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 140
EXPECTED_HOURS = 2.5
TIME_LIMIT_MINUTES = EXPECTED_HOURS * 60

# Setup do log
log_file = f"full_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_file,
    filemode='w',
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)

# Carregamento de dados
with open("arc-agi_test_challenges.json") as f:
    tasks = json.load(f)

train_start = time.time()
log("[INFO] Preparando dados do SageAxiom...")
X_train_all, y_train_all = [], []
for task in tasks.values():
    for pair in task["train"]:
        input_grid = pad_to_shape(tf.convert_to_tensor(pair["input"], dtype=tf.int32))
        output_grid = pad_to_shape(tf.convert_to_tensor(pair["output"], dtype=tf.int32))
        X_train_all.append(input_grid)
        y_train_all.append(output_grid)

X_all = tf.stack(X_train_all)
y_all = tf.stack(y_train_all)

# Verifica se algum valor está fora do vocabulário
max_value = tf.reduce_max(tf.maximum(tf.reduce_max(X_all), tf.reduce_max(y_all))).numpy()
assert max_value < VOCAB_SIZE, f"[ERRO] Valor fora do vocabulário detectado: {max_value} >= {VOCAB_SIZE}"

X_all_onehot = tf.one_hot(X_all, depth=VOCAB_SIZE)

X_train, X_val, y_train, y_val = train_test_split(
    X_all_onehot.numpy(), y_all.numpy(), test_size=0.2, random_state=42
)

X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)

# Treinamento
models = []
for i in range(NUMBER_OF_MODELS):
    log(f"[INFO] Iniciando treino do modelo SageAxiom_{i+1}...")
    model = SageAxiom(hidden_dim=128, use_hard_choice=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=SparseFocalLoss,
        metrics=["accuracy"]
    )

    os.makedirs(f"checkpoints/sage_axiom_{i+1}", exist_ok=True)
    callbacks = [
        ModelCheckpoint(f"checkpoints/sage_axiom_{i+1}/model", monitor="val_loss", save_best_only=True, save_format="tf", verbose=1),
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )

    dummy_input = tf.random.uniform((1, 30, 30, VOCAB_SIZE))
    dummy_text_embed = tf.random.uniform((1, 128))
    _ = model(dummy_input, text_embed=dummy_text_embed)

    model.save(f"sage_model_{i+1}", save_format="tf", save_traces=False)

    plot_history(history, i)
    y_val_pred = tf.argmax(model(X_val, training=False), axis=-1).numpy()
    plot_confusion(y_val.numpy(), y_val_pred, i)
    models.append(model)

profile_time(train_start, "[INFO] Tempo total de treinamento")
