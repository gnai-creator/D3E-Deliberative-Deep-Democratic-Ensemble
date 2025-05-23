import numpy as np

import numpy as np

def pad_to_shape(grid, target_shape=(30, 30), pad_value=0):
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)  # força conversão segura
    h, w = grid.shape
    H, W = target_shape
    padded = np.full(target_shape, pad_value, dtype=grid.dtype)
    padded[:h, :w] = grid
    return padded




def pad_to_shape_batch(grids, target_shape, pad_value=0):
    if isinstance(grids, np.ndarray) and grids.ndim == 2:
        grids = [grids]  # wrap single example as list
    padded_batch = [pad_to_shape(grid, target_shape, pad_value) for grid in grids]
    return np.stack(padded_batch)



# Example in main.py after checking shape mismatch:
# if Y_train.shape[:2] != pred_shape_hw:
#     Y_train = pad_to_shape_batch(Y_train, pred_shape_hw, pad_value=0)
#     Y_val = pad_to_shape_batch(Y_val, pred_shape_hw, pad_value=0)

def align_shapes(a, b, pad_value=0):
    H = max(a.shape[0], b.shape[0])
    W = max(a.shape[1], b.shape[1])
    def pad_to(x, H, W):
        padded = np.full((H, W), pad_value, dtype=x.dtype)
        padded[:x.shape[0], :x.shape[1]] = x
        return padded
    return pad_to(a, H, W), pad_to(b, H, W)