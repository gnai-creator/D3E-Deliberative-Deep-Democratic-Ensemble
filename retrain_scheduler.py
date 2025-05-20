# retrain_scheduler.py
import os
import json
import tensorflow as tf
from core import SageAxiom
from losses import masked_sparse_categorical_loss
from runtime_utils import log, pad_to_shape
from metrics_utils import plot_prediction_debug, plot_confusion, plot_history
from data_augmentation import augment_data

# Hyperparams
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 60

RESULTS_DIR = "results"
SCOREBOARD_PATH = "model_scoreboard.json"

def load_scoreboard():
    if os.path.exists(SCOREBOARD_PATH):
        with open(SCOREBOARD_PATH) as f:
            return json.load(f)
    return {}

def retrain_low_score_tasks(tasks, score_threshold=0.05):
    scoreboard = load_scoreboard()
    for task_id, scores in scoreboard.items():
        for model_name, acc in scores.items():
            if acc < score_threshold:
                model_index = int(model_name.split("_")[-1])
                log(f"[RETRAIN] Task {task_id} - {model_name} (acc={acc:.3f})")

                train_pairs = [aug for pair in tasks[task_id]["train"] for aug in augment_data(pair)]
                X_train, y_train = [], []

                for pair in train_pairs:
                    input_grid = pad_to_shape(tf.convert_to_tensor(pair["input"], dtype=tf.int32))
                    output_grid = pad_to_shape(tf.convert_to_tensor(pair["output"], dtype=tf.int32))
                    X_train.append(input_grid)
                    y_train.append(output_grid)

                X_all = tf.stack(X_train)
                y_all = tf.stack(y_train)

                X_onehot = tf.one_hot(X_all, depth=10)
                X_train_np = X_onehot.numpy()
                y_train_np = y_all.numpy()

                model = SageAxiom(hidden_dim=128)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    loss=masked_sparse_categorical_loss,
                    metrics=["accuracy"]
                )

                history = model.fit(
                    X_train_np, y_train_np,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1
                )

                model_dir = os.path.join(RESULTS_DIR, task_id)
                model_path = os.path.join(model_dir, f"{model_name}_retrained")
                model.save(model_path, save_format="tf", save_traces=False)

                plot_history(history, model_name=f"{task_id}_{model_name}_retrained")
                val_sample = tf.convert_to_tensor(X_train_np[:1], dtype=tf.float32)
                pred = tf.argmax(model(val_sample, training=False), axis=-1).numpy()
                plot_prediction_debug(val_sample[0], y_train_np[0], pred[0], f"{task_id}_{model_name}_retrained")
                log(f"[RETRAIN] Finalizado: {task_id} - {model_name}")