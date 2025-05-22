# data_augmentation.py

import numpy as np
import random
from copy import deepcopy

# --- ROTATIONS ---
def rotate_pair(pair, k=1):
    return {
        "input": np.rot90(pair["input"], k).tolist(),
        "output": np.rot90(pair["output"], k).tolist()
    }

# --- FLIPS ---
def flip_pair(pair):
    return {
        "input": np.fliplr(pair["input"]).tolist(),
        "output": np.fliplr(pair["output"]).tolist()
    }



# --- MAIN AUGMENTATION FUNCTION ---
def augment_dataset_rotation_flip(X_data, y_data):
    """
    Aplica rotação (90, 180, 270) e flip horizontal em cada par (x, y).

    Args:
        X_data (list or array): Lista de matrizes de entrada
        y_data (list or array): Lista de matrizes de saída

    Returns:
        X_aug (list): Entradas aumentadas
        y_aug (list): Saídas aumentadas
    """
    X_aug, y_aug = [], []

    for x, y in zip(X_data, y_data):
        pair = {"input": x, "output": y}
        augmented = [pair]
        # Rotations
        for k in range(1, 4):
            augmented.append(rotate_pair(pair, k))
        # Flip
        augmented.append(flip_pair(pair))

        for aug in augmented:
            X_aug.append(np.array(aug["input"]))
            y_aug.append(np.array(aug["output"]))

    return X_aug, y_aug


def augment_with_class_replacement(X_data, y_data, pad_value=-1, num_classes=10):
    """
    Para cada par (x, y) onde y contém pelo menos um zero (classe a ser augmentada),
    gera num_classes versões substituindo os 0s por [0, 1, ..., 9].

    Retorna: X_augmented, y_augmented (aumentados)
    """
    X_aug = []
    y_aug = []

    for x, y in zip(X_data, y_data):
        y = np.array(y)
        if 0 in y:
            for class_id in range(num_classes):
                y_augmented = np.where(y == 0, class_id, y)
                X_aug.append(np.copy(x))
                y_aug.append(y_augmented)
        else:
            X_aug.append(np.copy(x))
            y_aug.append(np.copy(y))
    
    return np.array(X_aug), np.array(y_aug)
