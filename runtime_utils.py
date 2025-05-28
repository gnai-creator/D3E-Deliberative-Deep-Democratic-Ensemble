# runtime_utils.py

import os
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import traceback
from sklearn.metrics import confusion_matrix, classification_report



log_filename = f"log_arc_{time.strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_filename,
    filemode='w',
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)

def log(msg):
    print(msg)
    logging.info(msg)

def make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj

def save_debug_result(data, filepath):
    serializable_data = make_serializable(data)
    with open(filepath, "w") as f:
        json.dump(serializable_data, f, indent=2)


def pad_to_shape(tensor, target_shape=(30, 30), pad_value=0):
    import tensorflow as tf

    tensor = tf.convert_to_tensor(tensor)

    # Se for 1D (ex: [0, 1, 2]), converte para 2D automaticamente
    if tensor.shape.rank == 1:
        tensor = tf.expand_dims(tensor, axis=0)  # (1, N)

    shape = tf.shape(tensor)
    rank = tensor.shape.rank

    if rank == 2:
        height, width = shape[0], shape[1]
        pad_height = tf.maximum(target_shape[0] - height, 0)
        pad_width = tf.maximum(target_shape[1] - width, 0)
        paddings = [[0, pad_height], [0, pad_width]]

    elif rank == 3:
        height, width, channels = shape[0], shape[1], shape[2]
        pad_height = tf.maximum(target_shape[0] - height, 0)
        pad_width = tf.maximum(target_shape[1] - width, 0)
        paddings = [[0, pad_height], [0, pad_width], [0, 0]]

    else:
        raise ValueError(f"[ERRO] Tensor com rank não suportado: {rank}")

    return tf.pad(tensor, paddings=paddings, constant_values=pad_value)

def profile_time(start, label):
    elapsed = time.time() - start
    mins, secs = divmod(elapsed, 60)
    log(f"[PERF] {label}: {int(mins)}m {int(secs)}s ({elapsed:.2f} segundos)")
    return elapsed

def ensure_batch_dim(tensor):
    return tf.expand_dims(tensor, axis=0) if tensor.shape.rank == 3 else tensor


def to_numpy_safe(x):
    return x.numpy() if isinstance(x, tf.Tensor) else np.array(x)