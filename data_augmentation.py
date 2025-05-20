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

# --- COLOR PERMUTATION ---
def color_permutation_pair(pair):
    input_arr = np.array(pair["input"])
    output_arr = np.array(pair["output"])
    unique_colors = np.unique(np.concatenate((input_arr.flatten(), output_arr.flatten())))
    permuted = np.random.permutation(unique_colors)
    mapping = dict(zip(unique_colors, permuted))
    input_mapped = np.vectorize(mapping.get)(input_arr)
    output_mapped = np.vectorize(mapping.get)(output_arr)
    return {
        "input": input_mapped.tolist(),
        "output": output_mapped.tolist()
    }

# --- TRANSLATION ---
def translate_pair(pair, dx=0, dy=0):
    def shift(grid):
        grid_np = np.array(grid)
        shifted = np.roll(grid_np, shift=(dy, dx), axis=(0, 1))
        if dy > 0:
            shifted[:dy, :] = 0
        elif dy < 0:
            shifted[dy:, :] = 0
        if dx > 0:
            shifted[:, :dx] = 0
        elif dx < 0:
            shifted[:, dx:] = 0
        return shifted.tolist()

    return {
        "input": shift(pair["input"]),
        "output": shift(pair["output"])
    }

# --- RANDOM CROP AND RESIZE ---
def random_crop_resize_pair(pair, crop_size=20, target_size=30):
    def process(grid):
        grid_np = np.array(grid)
        h, w = grid_np.shape
        if h < crop_size or w < crop_size:
            return grid_np
        x = random.randint(0, h - crop_size)
        y = random.randint(0, w - crop_size)
        cropped = grid_np[x:x + crop_size, y:y + crop_size]
        resized = np.zeros((target_size, target_size), dtype=grid_np.dtype)
        resized[:crop_size, :crop_size] = cropped
        return resized.tolist()

    return {
        "input": process(pair["input"]),
        "output": process(pair["output"])
    }

# --- MAIN AUGMENTATION FUNCTION ---
def augment_data(pair):
    augmented = [pair]
    # Rotations
    for k in range(1, 4):
        augmented.append(rotate_pair(pair, k))
    # Flip
    augmented.append(flip_pair(pair))
    # Color permutation
    augmented.append(color_permutation_pair(pair))
    # Translations (small shifts)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        augmented.append(translate_pair(pair, dx, dy))
    # Random crop & resize
    augmented.append(random_crop_resize_pair(pair))
    return augmented