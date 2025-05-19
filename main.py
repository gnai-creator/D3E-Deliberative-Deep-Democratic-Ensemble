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

from core import SageAxiom
from metrics_utils import (
    plot_history,
    plot_confusion,
    plot_attempts_stats,
    plot_prediction_debug,
    visualize_attention_map,
    MaskedIoU,
    plot_logit_distribution
)
from sage_debate_loop import conversational_loop
from runtime_utils import log, pad_to_shape, profile_time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from losses import dynamic_focal_loss_wrapper, AlphaWarmupCallback
# from model_improvements import compute_aggressive_class_weights, spatial_augmentations
from model_improvements import spatial_augmentations

VOCAB_SIZE = 10
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
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X_all_onehot.numpy(), y_all.numpy(), test_size=0.2, random_state=42
)

aug_X, aug_y = [], []
for x, y in zip(X_train_np, y_train_np):
    ax, ay = spatial_augmentations(tf.convert_to_tensor(x), tf.convert_to_tensor(y))
    aug_X.append(ax)
    aug_y.append(ay)
X_train = tf.stack(aug_X)
y_train = tf.stack(aug_y)

X_val = tf.convert_to_tensor(X_val_np, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val_np, dtype=tf.int32)

# class_weight_array = compute_aggressive_class_weights(y_train, 0.4)

models = []
for i in range(NUMBER_OF_MODELS):
    log(f"[INFO] Iniciando treino do modelo SageAxiom_{i+1}...")
    # alpha_var = tf.Variable(initial_value=class_weight_array, dtype=tf.float32, trainable=False)
    # warmup_cb = AlphaWarmupCallback(
    #     alpha_var=alpha_var,
    #     initial_alpha=class_weight_array,
    #     target_alpha=class_weight_array,
    #     warmup_epochs=10
    # )

    model = SageAxiom(hidden_dim=128)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        # loss=dynamic_focal_loss_wrapper(alpha_var=alpha_var, gamma=0.25),
        # metrics=["accuracy", MaskedIoU(num_classes=VOCAB_SIZE, ignore_class=0)]
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    os.makedirs(f"checkpoints/sage_axiom_{i+1}", exist_ok=True)
    callbacks = [
        ModelCheckpoint(f"checkpoints/sage_axiom_{i+1}/model", monitor="val_loss", save_best_only=True, save_format="tf", verbose=0),
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=FACTOR, patience=RL_PATIENCE, min_lr=1e-5, verbose=0),
        # warmup_cb
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )

    model.save(f"sage_model_{i}", save_format="tf", save_traces=False)
    plot_history(history, i)
    y_val_pred = tf.argmax(model(X_val, training=False), axis=-1).numpy()
    plot_confusion(y_val.numpy(), y_val_pred, i)
    models.append(model)

    sample_input = X_val[0:1]
    sample_target = y_val[0:1]
    pred_logits = model(sample_input, training=False)
    plot_logit_distribution(pred_logits[0], model_index=i)
    pred_output = tf.argmax(pred_logits[0], axis=-1).numpy()
    plot_prediction_debug(sample_input[0], sample_target[0], pred_output, f"val_sample_model_{i}")

    if hasattr(model, 'last_attention_output') and model.last_attention_output is not None:
        visualize_attention_map(model.last_attention_output, i, title=f"Attention Map - Model {i}")

elapsed_train_time = profile_time(train_start, "[INFO] Tempo total de treinamento")


# === Avaliação ===
submission_dict = defaultdict(list)
correct_tasks, total_tasks = 0, 0
scores, task_ids = [], []
task_times, attempts_per_task = {}, {}
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
        break

    input_grid = task["train"][0]["input"]
    result = conversational_loop(models, input_grid, max_rounds=10000)

    if result["success"]:
        correct_tasks += 1
    submission_dict[task_id] = [result["output"]] if result["output"] else []

    with open(f"history_prompts/{task_id}.json", "w") as f:
        json.dump(result["history"], f, indent=2)

    with open(f"history_prompts/{task_id}.md", "w", encoding="utf-8") as md:
        md.write(f"# Task {task_id}\n")
        for round_num, entry in enumerate(result["history"], 1):
            md.write(f"## Round {round_num}\n")
            for model_idx, candidate in enumerate(entry["candidates"]):
                md.write(f"### Modelo {model_idx+1}\n\n```python\n{json.dumps(candidate)}\n```\n")
            md.write(f"**Votos**: {entry['votes']}\n\n**Ganhador**: Modelo {entry['winner']}\n")
            model_vote_stats[f"model_{entry['winner']+1}"] += 1
            model_wins[f"model_{entry['winner']+1}"] += 1

    task_times[task_id] = profile_time(time.time(), f"Task {task_id}")
    attempts_per_task[task_id] = result["rounds"]
    total_tasks += 1
    task_ids.append(task_id)
    scores.append(int(result["success"]))

    if time.time() > end_time:
        break

with open("submission.json", "w") as f:
    json.dump(submission_dict, f)
with open("per_task_times.json", "w") as f:
    json.dump(task_times, f, indent=2)

plot_attempts_stats(task_times, attempts_per_task)
score = (correct_tasks / total_tasks) * 100 if total_tasks > 0 else 0
projection = (correct_tasks / 250) * 100

log(f"[INFO] Score estimado: {score:.2f}%")
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