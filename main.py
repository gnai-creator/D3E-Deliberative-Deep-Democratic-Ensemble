# main.py

import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
for handler in tf.get_logger().handlers:
    tf.get_logger().removeHandler(handler)

import logging
import json
import time
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from core import SageAxiom
from metrics_utils import (
    plot_history,
    plot_confusion,
    plot_attempts_stats,
    plot_prediction_debug
)
from sage_debate_loop import conversational_loop
from runtime_utils import log, pad_to_shape, profile_time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from losses import SparseFocalLoss

VOCAB_SIZE = 15
NUMBER_OF_MODELS = 5
LEARNING_RATE = 0.001
PATIENCE = 15
RL_PATIENCE = 5
FACTOR = 0.5
BATCH_SIZE = 16
EPOCHS = 140
EXPECTED_HOURS = 2.5
TIME_LIMIT_MINUTES = EXPECTED_HOURS * 60

log_file = f"full_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_file,
    filemode='w',
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)

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

y_train_flat = y_train.numpy().flatten()
classes = np.unique(y_train_flat)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_flat)
class_weight_array = np.ones(VOCAB_SIZE)
for cls, weight in zip(classes, class_weights):
    class_weight_array[cls] = weight

models = []
for i in range(NUMBER_OF_MODELS):
    log(f"[INFO] Iniciando treino do modelo SageAxiom_{i+1}...")
    model = SageAxiom(hidden_dim=128, use_hard_choice=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=SparseFocalLoss(gamma=2.0, alpha=class_weight_array),
        metrics=["accuracy"]
    )

    os.makedirs(f"checkpoints/sage_axiom_{i+1}", exist_ok=True)
    callbacks = [
        ModelCheckpoint(f"checkpoints/sage_axiom_{i+1}/model", monitor="val_loss", save_best_only=True, save_format="tf", verbose=0),
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=FACTOR, patience=RL_PATIENCE, min_lr=1e-5, verbose=0)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )

    model.save(f"sage_model_{i+1}", save_format="tf", save_traces=False)
    plot_history(history, i)
    y_val_pred = tf.argmax(model(X_val, training=False), axis=-1).numpy()
    plot_confusion(y_val.numpy(), y_val_pred, i)
    models.append(model)

    sample_input = X_val[0:1]
    sample_target = y_val[0:1]
    pred_logits = model(sample_input, training=False)
    pred_output = tf.argmax(pred_logits[0], axis=-1).numpy()
    plot_prediction_debug(sample_input[0], sample_target[0], pred_output, f"val_sample_model_{i+1}")

train_tasks = [(task_id, task) for task_id, task in tasks.items() if "train" in task]
for task_id, task in train_tasks:
    example = task["train"][0]
    input_grid = pad_to_shape(tf.convert_to_tensor(example["input"], dtype=tf.int32))
    target_grid = pad_to_shape(tf.convert_to_tensor(example["output"], dtype=tf.int32))
    input_tensor = tf.one_hot(input_grid, depth=VOCAB_SIZE)[None, ...]
    target_tensor = target_grid[None, ...]
    for idx, model in enumerate(models):
        model.fit(
            input_tensor,
            target_tensor,
            batch_size=1,
            epochs=1,
            verbose=1
        )
        pred_logits = model(input_tensor, training=False)
        pred_output = tf.argmax(pred_logits[0], axis=-1).numpy()
        plot_prediction_debug(input_tensor[0], target_tensor[0], pred_output, f"task_{task_id}_model_{idx+1}")

elapsed_train_time = profile_time(train_start, "[INFO] Tempo total de treinamento")

# Avaliação
submission_dict = defaultdict(list)
correct_tasks = 0
total_tasks = 0
task_times = {}
attempts_per_task = {}
scores = []
task_ids = []
model_vote_stats = {f"model_{i+1}": 0 for i in range(NUMBER_OF_MODELS)}
model_wins = Counter()

os.makedirs("history_prompts", exist_ok=True)

evaluation_start = time.time()
end_time = evaluation_start + (TIME_LIMIT_MINUTES * 60) - elapsed_train_time

task_iter = iter(tasks.items())
while time.time() < end_time:
    try:
        task_id, task = next(task_iter)
    except StopIteration:
        log("[INFO] Todas as tasks foram avaliadas.")
        break

    task_start = time.time()
    input_grid = task["train"][0]["input"]
    log(f"[INFO] Avaliando task {task_id} ({total_tasks + 1})")

    result = conversational_loop(models, input_grid, max_rounds=10000)

    if result["success"]:
        log(f"[INFO] Task {task_id} avaliada com sucesso.")
        log(f"[INFO] Saída correta para a task {task_id}: {result['output']}")
        correct_tasks += 1
    submission_dict[task_id] = [result["output"]] if result["output"] else []

    with open(f"history_prompts/{task_id}.json", "w") as f:
        json.dump(result["history"], f, indent=2)

    with open(f"history_prompts/{task_id}.md", "w", encoding="utf-8") as md:
        md.write(f"# Histórico da Task {task_id}\n\n")
        for round_num, entry in enumerate(result["history"], 1):
            md.write(f"## Rodada {round_num}\n")
            md.write(f"\n**Entradas**\n\n")
            for model_idx, candidate in enumerate(entry["candidates"]):
                md.write(f"### Modelo {model_idx+1}\n\n")
                md.write("```python\n")
                md.write(json.dumps(candidate) + "\n")
                md.write("```\n\n")
            md.write(f"**Votos**: {entry['votes']}\n\n")
            md.write(f"**Ganhador**: Modelo {entry['winner']}\n\n")
            model_vote_stats[f"model_{entry['winner']+1}"] += 1
            model_wins[f"model_{entry['winner']+1}"] += 1

    elapsed = profile_time(task_start, f"Task {task_id}")
    task_times[task_id] = elapsed
    attempts_per_task[task_id] = result["rounds"]
    total_tasks += 1
    task_ids.append(task_id)
    scores.append(int(result["success"]))

    if time.time() > end_time:
        log("[INFO] Tempo total atingido. Encerrando avaliação.")
        break

with open("submission.json", "w") as f:
    json.dump(submission_dict, f, ensure_ascii=False)

with open("per_task_times.json", "w") as f:
    json.dump(task_times, f, indent=2)

plot_attempts_stats(task_times, attempts_per_task)
log(f"[INFO] Matches corretos: {correct_tasks}/{total_tasks}")
score = (correct_tasks / total_tasks) * 100 if total_tasks > 0 else 0
log(f"[INFO] Score estimado: {score:.2f}%")
projection = (correct_tasks / 250) * 100
log(f"[INFO] Projeção final aproximada: {projection:.2f}%")
log(f"[INFO] Votos por modelo: {dict(model_vote_stats)}")
log(f"[INFO] Vitórias por modelo: {dict(model_wins)}")

hardest_tasks = sorted(task_times.items(), key=lambda x: x[1], reverse=True)[:5]
most_attempts = sorted(attempts_per_task.items(), key=lambda x: x[1], reverse=True)[:5]
log("[INFO] Tasks mais demoradas:")
for tid, duration in hardest_tasks:
    log(f" - {tid}: {duration:.2f} segundos")
log("[INFO] Tasks com mais tentativas:")
for tid, rounds in most_attempts:
    log(f" - {tid}: {rounds} rodadas")

log("[INFO] Pipeline encerrado.")