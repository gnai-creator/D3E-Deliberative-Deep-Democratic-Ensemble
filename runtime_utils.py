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


def pad_to_shape(tensor, target_shape=(30, 30), pad_value=-1):
    # Get dynamic shape
    shape = tensor.shape
    height = shape[0]
    width = shape[1]

    pad_height = target_shape[0] - height
    pad_width = target_shape[1] - width

    paddings = [[0, pad_height], [0, pad_width]]
    return tf.pad(tensor, paddings=paddings, constant_values=pad_value)



def profile_time(start, label):
    elapsed = time.time() - start
    mins, secs = divmod(elapsed, 60)
    log(f"[PERF] {label}: {int(mins)}m {int(secs)}s ({elapsed:.2f} segundos)")
    return elapsed

