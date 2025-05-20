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
from data_augmentation import augment_data

# --- Configs e Hiperparâmetros ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

VOCAB_SIZE = 10
LEARNING_RATE = 0.001
PATIENCE = 25
RL_PATIENCE = 5
RL_LEARNING_RATE = 1e-4
FACTOR = 0.5
BATCH_SIZE = 32
EPOCHS = 140
RESULTS_DIR = "results"
MODELS_PER_TASK = 3
N_TASKS = 1
HOURS = 4
MAX_TRAINING_TIME = HOURS * 60 * 60
MAX_EVAL_TIME = HOURS * 2 * 60 * 60 / (N_TASKS * MODELS_PER_TASK)

# --- Carregamento de Dados ---
with open("arc-agi_training_challenges.json") as f:
    train_challenges = json.load(f)
with open("arc-agi_training_solutions.json") as f:
    train_solutions = json.load(f)
with open("arc-agi_evaluation_challenges.json") as f:
    eval_challenges = json.load(f)
with open("arc-agi_evaluation_solutions.json") as f:
    eval_solutions = json.load(f)

train_challenges = dict(list(train_challenges.items())[:N_TASKS])
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAINING_TIME_LIMIT = MAX_TRAINING_TIME / (N_TASKS * MODELS_PER_TASK)
EVAL_TIME_LIMIT = MAX_EVAL_TIME / (N_TASKS * MODELS_PER_TASK)
log(f"[INFO] Tempo máximo de treinamento por task: {TRAINING_TIME_LIMIT:.2f} segundos")
log(f"[INFO] Tempo máximo de avaliação por task: {EVAL_TIME_LIMIT:.2f} segundos")
log(f"[INFO] Tempo máximo total de treinamento: {MAX_TRAINING_TIME:.2f} segundos")
log(f"[INFO] Tempo máximo total de avaliação: {MAX_EVAL_TIME:.2f} segundos")
log(f"[INFO] Número de tasks: {N_TASKS}")
log(f"[INFO] Número de modelos por task: {MODELS_PER_TASK}")
log(f"[INFO] Tamanho do vocabulário: {VOCAB_SIZE}")
log(f"[INFO] Tamanho do batch: {BATCH_SIZE}")
log(f"[INFO] Número de épocas: {EPOCHS}")
log(f"[INFO] Taxa de aprendizado: {LEARNING_RATE}")
log(f"[INFO] Paciência: {PATIENCE}")
log(f"[INFO] Paciência para redução de taxa de aprendizado: {RL_PATIENCE}")
log(f"[INFO] Fator de redução de taxa de aprendizado: {FACTOR}")
log(f"[INFO] Taxa mínima de aprendizado: {RL_LEARNING_RATE}")
log(f"[INFO] Diretório de resultados: {RESULTS_DIR}")
log(f"[INFO] Carregando dados de treinamento e avaliação...")
scores = {}
task_times = defaultdict(float)
submission_dict = {}
evaluation_logs = {}

start_time = time.time()
while (time.time() - start_time) < MAX_TRAINING_TIME:
    for task_id, task in train_challenges.items():
        if task_times[task_id] >= TRAINING_TIME_LIMIT:
            continue

        log(f"[INFO] Treinando modelos para task: {task_id}")
        train_pairs = [aug for aug in augment_data({
            "input": task["train"][0]["input"],
            "output": train_solutions[task_id][0]
        })]

        X_train, y_train = [], []
        for pair in train_pairs:
            input_grid = pad_to_shape(tf.convert_to_tensor(pair["input"], dtype=tf.int32))
            output_grid = pad_to_shape(tf.convert_to_tensor(pair["output"], dtype=tf.int32))
            X_train.append(input_grid)
            y_train.append(output_grid)

        X_all = tf.stack(X_train)
        y_all = tf.stack(y_train)
        X_onehot = tf.one_hot(X_all, depth=VOCAB_SIZE)

        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            X_onehot.numpy(), y_all.numpy(), test_size=0.2, random_state=42)

        X_val = tf.convert_to_tensor(X_val_np, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val_np, dtype=tf.int32)

        model_dir = os.path.join(RESULTS_DIR, task_id)
        os.makedirs(model_dir, exist_ok=True)
        models = []

        for i in range(MODELS_PER_TASK):
            model_path = os.path.join(model_dir, f"model_{i}")
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={"SageAxiom": SageAxiom, "masked_sparse_categorical_loss": masked_sparse_categorical_loss}
                )
                log(f"[INFO] Modelo carregado de {model_path}")
            except (IOError, OSError, ValueError):
                model = SageAxiom(hidden_dim=256)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss=masked_sparse_categorical_loss,
                    metrics=["accuracy"]
                )

                callbacks = [
                    EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=FACTOR, patience=RL_PATIENCE, min_lr=RL_LEARNING_RATE),
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
                elapsed = profile_time(task_start, f"[TEMPO] Task {task_id} - Modelo {i}")
                task_times[task_id] += elapsed

                plot_history(history, model_name=f"{task_id}_model_{i}")
                y_val_pred = tf.argmax(model(X_val, training=False), axis=-1).numpy()
                plot_confusion(y_val.numpy(), y_val_pred, model_name=f"{task_id}_model_{i}")
                plot_prediction_debug(X_val[0], y_val[0], y_val_pred[0], f"{task_id}_model_{i}")

            models.append(model)

        # Avaliação intermediária com solução de treino
        expected = train_solutions[task_id][0]
        input_grid = task["train"][0]["input"]
        result = conversational_loop(models, input_grid, max_rounds=10)
        predicted = result.get("output")
        scores[task_id] = (predicted == expected)
        evaluation_logs[task_id] = result
        submission_dict[task_id] = [predicted] if predicted else []

# Avaliação final com eval_challenges
log("[INFO] Avaliação final")
eval_start_time = time.time()
while (time.time() - eval_start_time) < EVAL_TIME_LIMIT:
    for task_id, task in list(eval_challenges.items())[:N_TASKS]:
        expected = eval_solutions[task_id]["solution"]
        models = []
        model_dir = os.path.join(RESULTS_DIR, task_id)

        for i in range(MODELS_PER_TASK):
            model_path = os.path.join(model_dir, f"model_{i}")
            try:
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={"SageAxiom": SageAxiom, "masked_sparse_categorical_loss": masked_sparse_categorical_loss}
                )
                models.append(model)
            except:
                continue

        if not models:
            log(f"[ERRO] Nenhum modelo válido para task {task_id}.")
            continue

        input_grid = task["test"][0]["input"]
        result = conversational_loop(models, input_grid, max_rounds=10)
        predicted = result.get("output")
        evaluation_logs[f"test_{task_id}"] = result
        submission_dict[task_id] = [predicted] if predicted else []
        log(f"[EVAL] {task_id}: {'✅' if predicted == expected else '❌'}")
    break

# Salvar resultados
with open("submission.json", "w") as f:
    json.dump(submission_dict, f, indent=2)
with open("evaluation_logs.json", "w") as f:
    json.dump(evaluation_logs, f, indent=2)

log("[RESULTADO FINAL] Avaliação completa.")