import os
import json
import time
import warnings
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import math
from core import SageAxiom
from metrics_utils import plot_history, plot_confusion, plot_prediction_debug
from runtime_utils import log, pad_to_shape, profile_time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sage_debate_loop import conversational_loop
from losses import masked_sparse_categorical_loss
from data_augmentation import augment_data

PAD_VALUE = -1

def masked_loss_with_smoothing(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, PAD_VALUE), tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-6)

# --- Configs e Hiperparâmetros ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

VOCAB_SIZE = 10
LEARNING_RATE = 0.0005
PATIENCE = 10
RL_PATIENCE = 3
RL_LEARNING_RATE = 3e-3
FACTOR = 0.7
BATCH_SIZE = 8
EPOCHS = 60
RESULTS_DIR = "results"
MODELS_PER_TASK = 3
BLOCK_SIZE = 400
MAX_BLOCKS = 1
HOURS = 1
MAX_TRAINING_TIME = HOURS * 60 * 60
MAX_EVAL_TIME = HOURS * 2 * 60 * 60 / MODELS_PER_TASK

# --- Carregamento de Dados ---
with open("arc-agi_training_challenges.json") as f:
    train_challenges = json.load(f)
with open("arc-agi_training_solutions.json") as f:
    train_solutions = json.load(f)
with open("arc-agi_evaluation_challenges.json") as f:
    eval_challenges = json.load(f)
with open("arc-agi_evaluation_solutions.json") as f:
    eval_solutions = json.load(f)

os.makedirs(RESULTS_DIR, exist_ok=True)

log(f"[INFO] Configurações gerais:")
log(f"[INFO] Tempo máximo total de treinamento: {MAX_TRAINING_TIME:.2f} segundos")
log(f"[INFO] Tempo máximo total de avaliação: {MAX_EVAL_TIME:.2f} segundos")
log(f"[INFO] Número de modelos por task: {MODELS_PER_TASK}")
log(f"[INFO] Tamanho do vocabulário: {VOCAB_SIZE}")
log(f"[INFO] Tamanho do batch: {BATCH_SIZE}")
log(f"[INFO] Número de épocas: {EPOCHS}")
log(f"[INFO] Taxa de aprendizado: {LEARNING_RATE}")
log(f"[INFO] Paciência: {PATIENCE}")
log(f"[INFO] Diretório de resultados: {RESULTS_DIR}")

scores = {}
submission_dict = {}
evaluation_logs = {}
task_ids = list(train_challenges.keys())
task_blocks = [task_ids[i:i + BLOCK_SIZE] for i in range(0, len(task_ids), BLOCK_SIZE)]

start_time = time.time()
block_index = 0
if task_blocks:
    block = task_blocks[0]
    if time.time() - start_time < MAX_TRAINING_TIME:
        log(f"[INFO] Treinando bloco {block_index} com tasks: {block}")
        X_train, y_train = [], []

        for task_id in block:
            solutions_list = train_solutions.get(task_id, [])
            num_pairs = min(len(train_challenges[task_id]["train"]), len(solutions_list))

            if num_pairs == 0:
                log(f"[WARN] Ignorando task {task_id}: sem pares válidos de treino.")
                continue

            for pair_idx in range(num_pairs):
                input_grid = train_challenges[task_id]["train"][pair_idx]["input"]
                output_grid = solutions_list[pair_idx]
                train_pairs = augment_data({"input": input_grid, "output": output_grid})

                for pair in train_pairs:
                    x = pad_to_shape(tf.convert_to_tensor(pair["input"], dtype=tf.int32))
                    y = pad_to_shape(tf.convert_to_tensor(pair["output"], dtype=tf.int32))
                    X_train.append(x)
                    y_train.append(y)

        if X_train:
            X_all = tf.stack(X_train)
            y_all = tf.stack(y_train)
            X_onehot = tf.one_hot(X_all, depth=VOCAB_SIZE)

            X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
                X_onehot.numpy(), y_all.numpy(), test_size=0.2, random_state=42)

            X_val = tf.convert_to_tensor(X_val_np, dtype=tf.float32)
            y_val_clean = np.where(y_val_np == -1, 0, y_val_np)
            val_sample_weight = (y_val_np != -1).astype(np.float32)
            y_val = tf.convert_to_tensor(y_val_clean, dtype=tf.int32)

            models = []
            oscillation = math.sin(block_index)
            scaling = 1 + 0.5 * oscillation
            learning_rate = LEARNING_RATE * scaling
            rl_learning_rate = RL_LEARNING_RATE * scaling

            for i in range(MODELS_PER_TASK):
                if time.time() - start_time >= MAX_TRAINING_TIME:
                    break
                model_dir = os.path.join(RESULTS_DIR, f"block_{block_index}")
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"model_{i}")
                model = SageAxiom(hidden_dim=256)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss=masked_loss_with_smoothing,
                    metrics=["accuracy"]
                )

                y_train_clean = np.where(y_train_np == -1, 0, y_train_np)
                sample_weight = (y_train_np != -1).astype(np.float32)

                task_start = time.time()
                log(f"[INFO] Treinando modelo {i} do bloco {block_index} com taxa de aprendizado {learning_rate:.6f}")
                # log de X_val shape
                log(f"[INFO] X_val shape: {X_val.shape}")
                # log X_train_np shape
                log(f"[INFO] X_train_np shape: {X_train_np.shape}")
                history = model.fit(
                    X_train_np, y_train_clean,
                    validation_data=(X_val, y_val, val_sample_weight),
                    sample_weight=sample_weight,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    callbacks=[
                        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
                        ReduceLROnPlateau(monitor="val_loss", factor=FACTOR, patience=RL_PATIENCE, min_lr=rl_learning_rate)
                    ]
                )

                model.save(model_path, save_format="tf", save_traces=False)
                profile_time(task_start, f"[TEMPO] Bloco {block_index} - Modelo {i}")

                plot_history(history, model_name=f"block_{block_index}_model_{i}")
                y_val_pred = tf.argmax(model(X_val, training=False), axis=-1).numpy()
                plot_confusion(y_val.numpy(), y_val_pred, model_name=f"block_{block_index}_model_{i}")
                plot_prediction_debug(X_val[0], y_val[0], y_val_pred[0], f"block_{block_index}_model_{i}")

                flat_true = y_val.numpy().flatten()
                flat_pred = y_val_pred.flatten()
                mask = flat_true != -1
                log("\n" + classification_report(flat_true[mask], flat_pred[mask], zero_division=0))
                log(f"[INFO] F1-score macro: {f1_score(flat_true[mask], flat_pred[mask], average='macro'):.4f}")

                models.append(model)

# Avaliação com MAX_BLOCKS
log("[INFO] Avaliação final")
eval_start_time = time.time()
eval_task_ids = list(eval_challenges.keys())
eval_blocks = [eval_task_ids[i:i + MAX_BLOCKS] for i in range(0, len(eval_task_ids), MAX_BLOCKS)]

def load_models():
    models = []
    for block_index in range(len(task_blocks)):
        model_dir = os.path.join(RESULTS_DIR, f"block_{block_index}")
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
    return models

for eval_block in eval_blocks:
    if time.time() - eval_start_time > MAX_EVAL_TIME:
        break

    models = load_models()
    for task_id in eval_block:
        if time.time() - eval_start_time > MAX_EVAL_TIME:
            break
        expected = eval_solutions[task_id]["solution"]
        input_grid = eval_challenges[task_id]["test"][0]["input"]
        result = conversational_loop(models, input_grid, max_rounds=10)
        predicted = result.get("output")
        evaluation_logs[f"test_{task_id}"] = result
        submission_dict[task_id] = [predicted] if predicted else []
        log(f"[EVAL] {task_id}: {'✅' if predicted == expected else '❌'}")

with open("submission.json", "w") as f:
    json.dump(submission_dict, f, indent=2)
with open("evaluation_logs.json", "w") as f:
    json.dump(evaluation_logs, f, indent=2)

log("[RESULTADO FINAL] Avaliação completa.")
log(f"[RESULTADO FINAL] Tempo total de execução: {time.time() - start_time:.2f} segundos")
log(f"[RESULTADO FINAL] Resultados: {json.dumps(scores, indent=2)}")
