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
from runtime_utils import log, pad_to_shape, profile_time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sage_debate_loop import conversational_loop
from losses import masked_loss_with_smoothing
from data_augmentation import augment_data
from data_preparation import get_dataset

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
while block_index * BLOCK_SIZE < len(task_ids) and time.time() - start_time < MAX_TRAINING_TIME:
    
    X_all, y_all, X_test = get_dataset(block_index, BLOCK_SIZE, task_ids, train_challenges)
    log(f"X_all: {X_all}")
    log(f"y_all: {y_all}")
    log(f"X_test: {X_test}")
        
    log(f"block_index: {block_index}")
    block_index += 1
    break

