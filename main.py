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
import tensorflow.keras as keras
from metrics_utils import plot_history, plot_confusion, plot_prediction_debug, plot_raw_input_preview
from runtime_utils import log, profile_time, ensure_batch_dim
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sage_debate_loop import conversational_loop
from losses import masked_loss_with_smoothing
from data_preparation import get_dataset
from model_compile import model_compilation

PAD_VALUE = -1

# Configurações de performance do TF

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_experimental_options({
    "layout_optimizer": False,
    "remapping": False,
    "constant_folding": True
})

# Hiperparâmetros
PAD_VALUE = -1
VOCAB_SIZE = 10
LEARNING_RATE = 0.0005
PATIENCE = 10
RL_PATIENCE = 3
RL_LEARNING_RATE = 3e-3
FACTOR = 0.7
BATCH_SIZE = 8
EPOCHS = 60
RESULTS_DIR = "results"
MODELS_PER_TASK = 1
BLOCK_SIZE = 1
LEN_TRAINING = 30
HOURS = 1
MAX_TRAINING_TIME = HOURS * 60 * 60
MAX_EVAL_TIME = HOURS * 2 * 60 * 60 / MODELS_PER_TASK

# Carregar dados
with open("arc-agi_training_challenges.json") as f:
    train_challenges = json.load(f)
with open("arc-agi_training_solutions.json") as f:
    train_solutions = json.load(f)
with open("arc-agi_evaluation_challenges.json") as f:
    eval_challenges = json.load(f)
with open("arc-agi_evaluation_solutions.json") as f:
    eval_solutions = json.load(f)

os.makedirs(RESULTS_DIR, exist_ok=True)

# Preparar tasks
task_ids = list(train_challenges.keys())
MAX_BLOCKS = len(task_ids)/ LEN_TRAINING

start_time = time.time()
scores = {}
submission_dict = {}
evaluation_logs = {}


# TREINAMENTO
for i in range(LEN_TRAINING):
    block_index = 0
    while block_index < MAX_BLOCKS and time.time() - start_time < MAX_TRAINING_TIME:
        
        if time.time() - start_time > MAX_TRAINING_TIME:
            break


        X_train_final, X_val_final, y_train_final, y_val_final, sw_train, sw_val, X_test_final, X_raw = get_dataset(block_index, task_ids, train_challenges, BLOCK_SIZE, PAD_VALUE, VOCAB_SIZE)
        if len(X_raw) == 0 or len(y_val_final) == 0:
            log(f"[WARN] Nenhuma amostra de validação disponível no bloco {block_index}")
            block_index += 1
            continue

        print(f"X_train_final shape: {X_train_final.shape}")
        plot_raw_input_preview(X_raw[0], model_name=f"block_{block_index}_task_{task_ids[block_index]}_input")

        oscillation = math.sin(block_index)
        scaling = 1 + 0.5 * oscillation
        learning_rate = LEARNING_RATE * scaling
        rl_learning_rate = RL_LEARNING_RATE * scaling
        model_path = ""
        models = []
        for i in range(MODELS_PER_TASK):
            if time.time() - start_time > MAX_TRAINING_TIME:
                break

            model, model_path = model_compilation(
                index=i,
                learning_rate=learning_rate,
                vocab_size=VOCAB_SIZE,
                block_index=block_index,
                result_dir=RESULTS_DIR
            )

            models.append(model)
        y_val_pred = tf.ones([1,30,30,10], dtype=tf.int64)
        for i in range(MODELS_PER_TASK):
            # while not tf.reduce_all(tf.equal(
            #     tf.cast(tf.argmax(y_val_pred, axis=-1), dtype=tf.int64),
            #     tf.cast(tf.gather(y_val_final, block_index), dtype=tf.int64)
            # )):

            # Processa input de treino
            x = tf.gather(X_train_final, 0)  # [30, 30, 10]
            x = tf.transpose(x, perm=[2, 0, 1])         # [T, H, W]
            x = tf.expand_dims(x, -1)                   # [T, H, W, C=1]
            x = tf.expand_dims(x, 0)                    # [B=1, T, H, W, C]
            x = tf.transpose(x, perm=[0, 2, 3, 1, 4])   # [B, H, W, T, C]

            # Processa input de validação (todo o batch de uma vez)
            def transform_input(x_sample):
                x_sample = tf.transpose(x_sample, perm=[2, 0, 1])       # [T, H, W]
                x_sample = tf.expand_dims(x_sample, -1)                 # [T, H, W, C=1]
                return tf.transpose(x_sample, perm=[1, 2, 0, 3])        # [H, W, T, C]

            X_val_transformed = tf.map_fn(transform_input, X_val_final)  # [B, H, W, T, C]

            # Targets e pesos
            y = tf.gather(y_train_final, 0)
            sw = tf.gather(sw_train, 0)

            log(f"Modelo {i} vai treinar a task {task_ids[block_index]}")
            log(f"[DEBUG] Shape x: {x.shape} | y: {y.shape} | sw: {sw.shape}")

            # Treinamento
            history = models[i].fit(
                x=x,  # [1, 30, 30, 10, 1]
                y={"main_output": tf.expand_dims(y, 0), "aux_output": tf.expand_dims(y, 0)}, # [1, 30, 30]

                sample_weight=tf.expand_dims(sw, 0),  # [1, 30, 30]
                validation_data=(
                    X_val_transformed,
                    (y_val_final, y_val_final),  # [1, 30, 30]
                    sw_val
                ),
                batch_size=1,
                epochs=EPOCHS,
                verbose=1,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=FACTOR, patience=RL_PATIENCE, min_lr=rl_learning_rate)
                ]
            )

            
            models[i].save_weights(model_path + "_weights.h5")
            plot_history(history, model_name=f"block_{block_index}_task_{task_ids[block_index]}_model_{i}")

            try:
                # Preparação do input
                x_val = tf.gather(X_val_final, 0)                  # [30, 30, 10]
                x_val = tf.transpose(x_val, perm=[2, 0, 1])        # [T, H, W]
                x_val = tf.expand_dims(x_val, -1)                  # [T, H, W, 1]
                x_val = tf.expand_dims(x_val, 0)                   # [1, T, H, W, 1]
                x_val = tf.transpose(x_val, perm=[0, 2, 3, 1, 4])  # [1, H, W, T, C]

                # Predição segura
                preds = models[i].predict(x_val)

                # Acessa main_output corretamente (dict ou tuple ou tensor direto)
                if isinstance(preds, dict):
                    y_val_logits = preds.get("main_output", None)
                    if y_val_logits is None:
                        raise ValueError("[ERROR] 'main_output' não encontrado nas predições.")
                elif isinstance(preds, (tuple, list)):
                    y_val_logits = preds[0]
                else:
                    y_val_logits = preds

                # Checagem do tipo antes do argmax
                if isinstance(y_val_logits, tf.Tensor) and y_val_logits.dtype == tf.string:
                    raise TypeError("[ERROR] y_val_logits do tipo string. Isso não deveria acontecer.")
                if isinstance(y_val_logits, np.ndarray) and y_val_logits.dtype.kind in {'U', 'S', 'O'}:
                    raise TypeError("[ERROR] y_val_logits contém strings ou objetos inválidos.")

                # Aplicar argmax com segurança
                y_val_pred = tf.argmax(y_val_logits, axis=-1).numpy()  # [1, H, W]

                # Debug dos valores únicos preditos
                unique_preds = np.unique(y_val_pred)
                log(f"[DEBUG] Valores únicos preditos: {unique_preds}")

                # Plot confusion
                y_val_expected = tf.gather(y_val_final, 0).numpy()
                plot_confusion(
                    y_val_expected,
                    y_val_pred[0],
                    model_name=f"block_{block_index}_task_{task_ids[block_index]}_model_{i}_index_{i}"
                )

                # Debug de shapes
                log(f"[DEBUG] Shapes => X_raw: {np.shape(X_raw)}, y_val_final: {np.shape(y_val_final)}, y_val_pred: {np.shape(y_val_pred)}")

                # Debug visual
                plot_prediction_debug(
                    input_tensor=X_raw[0],
                    expected_output=y_val_expected,
                    predicted_output=y_val_pred[0],
                    model_index=f"block_{block_index}_task_{task_ids[block_index]}_model_{i}",
                    index=i
                )

            except Exception as e:
                log(f"[ERROR] Erro ao gerar predicoes: {e}")


        block_index += 1 
    # printar tempo de treinamento
    # elapsed =profile_time(start_time, f"Tempo de treinamento finalizado")
    # log(f"Tempo de Treinamento : {elapsed}")
    # # DEBATE
    # while block_index * BLOCK_SIZE < len(task_ids) and time.time() - start_time < MAX_TRAINING_TIME:
        

            
    #     log(f"block_index: {block_index}")
    #     block_index += 1
    #     break

