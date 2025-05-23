import numpy as np

def pad_to_shape(grid, target_shape=(30, 30), pad_value=0):
    h, w = grid.shape
    H, W = target_shape
    padded = np.full(target_shape, pad_value, dtype=grid.dtype)
    padded[:h, :w] = grid
    return padded
