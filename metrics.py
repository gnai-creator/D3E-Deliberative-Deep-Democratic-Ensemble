import numpy as np

def pad_to_shape(grid, target_shape=(30, 30), pad_value=0):
    h, w = grid.shape
    H, W = target_shape
    padded = np.full(target_shape, pad_value, dtype=grid.dtype)
    padded[:h, :w] = grid
    return padded

def align_shapes(a, b, pad_value=0):
    H = max(a.shape[0], b.shape[0])
    W = max(a.shape[1], b.shape[1])
    def pad_to(x, H, W):
        padded = np.full((H, W), pad_value, dtype=x.dtype)
        padded[:x.shape[0], :x.shape[1]] = x
        return padded
    return pad_to(a, H, W), pad_to(b, H, W)