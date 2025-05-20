import os
import json
import time
import warnings
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

from core import SageAxiom
from metrics_utils import plot_history, plot_confusion, plot_prediction_debug
from runtime_utils import log, pad_to_shape, profile_time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sage_debate_loop import conversational_loop
from losses import masked_sparse_categorical_loss

# Silenciar avisos
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Carregar tasks e calcular limites
with open("arc-agi_test_challenges.json") as f:
    tasks = json.load(f)
num_tasks = len(tasks)
log(f"[INFO] Total de tasks: {num_tasks}")

# Configs
VOCAB_SIZE = 10
LEARNING_RATE = 0.001
PATIENCE = 10
RL_PATIENCE = 5
FACTOR = 0.5
BATCH_SIZE = 16
EPOCHS = 140
RESULTS_DIR = "results"
MODELS_PER_TASK = 5
HOURS = 4
MAX_TRAINING_TIME = HOURS * 60 * 60
MAX_EVAL_TIME = (12 - HOURS) * 60 * 60

TRAINING_TIME_LIMIT = MAX_TRAINING_TIME / (num_tasks * MODELS_PER_TASK)
EVAL_TIME_LIMIT = MAX_EVAL_TIME / (num_tasks * MODELS_PER_TASK)
log(f"[INFO] Tempo máximo de treinamento por modelo: {TRAINING_TIME_LIMIT:.2f} segundos")
log(f"[INFO] Tempo máximo de avaliação por modelo: {EVAL_TIME_LIMIT:.2f} segundos")

start_time = time.time()
os.makedirs(RESULTS_DIR, exist_ok=True)

scores = {}
task_times = defaultdict(float)
submission_dict = {}
evaluation_logs = {}

while (time.time() - start_time) < MAX_TRAINING_TIME:
    for task_id, task in tasks.items():
        if not task.get("train"):
            continue

        if task_times[task_id] >= TRAINING_TIME_LIMIT:
            log(f"[INFO] Task {task_id} já atingiu o limite de tempo de treino. Pulando.")
            continue

        log(f"[INFO] Treinando modelos para task: {task_id}")
        X_train, y_train = [], []
        for pair in task["train"]:
            X_train.append(pad_to_shape(tf.convert_to_tensor(pair["input"], dtype=tf.int32)))
            y_train.append(pad_to_shape(tf.convert_to_tensor(pair["output"], dtype=tf.int32)))

        X_all = tf.stack(X_train)
        y_all = tf.stack(y_train)

        assert tf.reduce_max(y_all).numpy() < VOCAB_SIZE
        X_onehot = tf.one_hot(X_all, depth=VOCAB_SIZE)
        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X_onehot.numpy(), y_all.numpy(), test_size=0.2)
        X_val, y_val = tf.convert_to_tensor(X_val_np, tf.float32), tf.convert_to_tensor(y_val_np, tf.int32)

        model_dir = os.path.join(RESULTS_DIR, task_id)
        os.makedirs(model_dir, exist_ok=True)
        models = []

        for i in range(MODELS_PER_TASK):
            model_path = os.path.join(model_dir, f"model_{i}")
            try:
                model = tf.keras.models.load_model(model_path, custom_objects={"SageAxiom": SageAxiom})
                log(f"[INFO] Modelo carregado de {model_path}")
            except:
                model = SageAxiom(hidden_dim=32)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss=masked_sparse_categorical_loss,
                    metrics=["accuracy"]
                )
                callbacks = [
                    EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=FACTOR, patience=RL_PATIENCE, min_lr=1e-5)
                ]
                task_start = time.time()
                history = model.fit(
                    X_train_np, y_train_np,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    callbacks=callbacks
                )
                model.save(model_path, save_format="tf", save_traces=False)
                task_times[task_id] += profile_time(task_start, f"[TEMPO] Task {task_id} - Modelo {i}")
                plot_history(history, model_name=f"{task_id}_model_{i}")
                y_val_pred = tf.argmax(model(X_val, training=False), axis=-1).numpy()
                plot_confusion(y_val.numpy(), y_val_pred, model_name=f"{task_id}_model_{i}")
                plot_prediction_debug(X_val[0], y_val[0], y_val_pred[0], f"{task_id}_model_{i}")
            models.append(model)

        y_val_pred = tf.argmax(models[0](X_val, training=False), axis=-1).numpy()
        acc = (y_val_pred == y_val.numpy()).mean()
        scores[task_id] = acc

        input_grid = task["train"][0]["input"]
        result = conversational_loop(models, input_grid, max_rounds=10)
        submission_dict[task_id] = [result["output"]] if result["output"] else []
        evaluation_logs[task_id] = result
        log(f"[INFO] Task {task_id} avaliada com sucesso")

log("[INFO] Avaliação final")
eval_start_time = time.time()
while (time.time() - eval_start_time) < EVAL_TIME_LIMIT:
    for task_id, task in tasks.items():
        if not task.get("test"):
            continue

        model_dir = os.path.join(RESULTS_DIR, task_id)
        models = []
        for i in range(MODELS_PER_TASK):
            model_path = os.path.join(model_dir, f"model_{i}")
            try:
                model = tf.keras.models.load_model(model_path, custom_objects={"SageAxiom": SageAxiom})
                models.append(model)
            except:
                log(f"[ERRO] Modelo model_{i} não encontrado para task {task_id}.")

        if not models:
            continue

        input_grid = task["test"][0]["input"]
        result = conversational_loop(models, input_grid, max_rounds=10)
        evaluation_logs[f"test_{task_id}"] = result
        log(f"[INFO] Avaliação de teste finalizada para task {task_id}")
    break

with open("submission.json", "w") as f:
    json.dump(submission_dict, f, indent=2)
with open("evaluation_logs.json", "w") as f:
    json.dump(evaluation_logs, f, indent=2)

sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for task_id, acc in sorted_scores:
    log(f"[SCORE] {task_id}: {acc:.3f}")

media = sum(scores.values()) / len(scores)
log(f"[RESULTADO FINAL] Média de acurácia por task: {media:.3f}")
total_train_time = sum(task_times.values())
log(f"[TEMPO TOTAL] Treinamento em {total_train_time:.2f} segundos")
