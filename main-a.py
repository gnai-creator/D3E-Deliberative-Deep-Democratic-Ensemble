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
from metrics_utils import plot_history, plot_confusion, plot_prediction_debug
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
MODELS_PER_TASK = 3
BLOCK_SIZE = 5
MAX_BLOCKS = 400 / BLOCK_SIZE
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

start_time = time.time()
scores = {}
submission_dict = {}
evaluation_logs = {}

block_index = 0


# TREINAMENTO
while block_index < len(task_ids) and time.time() - start_time < MAX_TRAINING_TIME:
    
    if time.time() - start_time > MAX_TRAINING_TIME:
        break


    X_train_final, X_val_final, y_train_final, y_val_final, sw_train, sw_val, X_test_final = get_dataset(block_index, task_ids, train_challenges, BLOCK_SIZE, PAD_VALUE, VOCAB_SIZE)
    
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
        while not tf.reduce_all(tf.equal(
            tf.cast(tf.argmax(y_val_pred, axis=-1), dtype=tf.int64),
            tf.cast(tf.gather(y_val_final, block_index), dtype=tf.int64)
        )):

            x = tf.gather(X_train_final, block_index)
            y = tf.gather(y_train_final, block_index)
            sw = tf.gather(sw_train, block_index)
            log(f"Modelo {i} vai treinar a task {task_ids[block_index]}")
            history = models[i].fit(
                x=tf.expand_dims(x, 0),  # batchify: (30, 30, 10) → (1, 30, 30, 10)
                y=tf.expand_dims(y, 0),  # (30, 30) → (1, 30, 30)
                sample_weight=tf.expand_dims(sw, 0),  # (30, 30) → (1, 30, 30)
                validation_data=(
                    X_val_final, y_val_final, sw_val
                ),
                batch_size=1,  # só uma amostra
                epochs=EPOCHS,
                verbose=1,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=FACTOR, patience=RL_PATIENCE, min_lr=rl_learning_rate)
                ]
            )
            
            models[i].save_weights(model_path + f"_task_{task_ids[block_index]}_weights.h5")
            plot_history(history, model_name=f"block_{block_index}_task_{task_ids[block_index]}_model_{i}")

            try:
                X_val = ensure_batch_dim(tf.gather(X_val_final, block_index))
                y_val_pred = models[i].predict(X_val)  # (batch, height, width, vocab)
                
                unique_preds = np.unique(np.argmax(y_val_pred, axis=-1))
                log(f"[DEBUG] Valores únicos preditos: {unique_preds}")

                # Reduz para classe predita por posição
                y_val_pred = tf.argmax(y_val_pred, axis=-1).numpy()  # (batch, height, width)

                plot_confusion(
                    y_val_final.numpy(),
                    y_val_pred,
                    model_name=f"block_{block_index}_task_{task_ids[block_index]}_model_{i}_index_{i}"
                )        

                plot_prediction_debug(
                    input_tensor=tf.gather(X_val_final, block_index),
                    expected_output=tf.gather(y_val_final, block_index),
                    predicted_output=y_val_pred[0],
                    model_index=f"block_{block_index}_task_{task_ids[block_index]}_model_{i}",
                    index=i
                )

            except Exception as e:
                log(f"[ERROR] Erro ao gerar predicoes: {e}")

    block_index += 1 
# printar tempo de treinamento
elapsed =profile_time(start_time, f"Tempo de treinamento finalizado")
log(f"Tempo de Treinamento : {elapsed}")
# DEBATE
while block_index * BLOCK_SIZE < len(task_ids) and time.time() - start_time < MAX_TRAINING_TIME:
    

        
    log(f"block_index: {block_index}")
    block_index += 1
    break

