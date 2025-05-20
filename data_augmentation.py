import numpy as np
import random

def rotate_pair(pair, k=1):
    return {
        "input": np.rot90(pair["input"], k).tolist(),
        "output": np.rot90(pair["output"], k).tolist()
    }

def flip_pair(pair):
    return {
        "input": np.fliplr(pair["input"]).tolist(),
        "output": np.fliplr(pair["output"]).tolist()
    }

def augment_data(pair):
    augmented = [pair]
    for k in range(1, 4):  # 90, 180, 270 degrees
        augmented.append(rotate_pair(pair, k))
    augmented.append(flip_pair(pair))
    return augmented
