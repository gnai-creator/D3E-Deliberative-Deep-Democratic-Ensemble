import numpy as np
import tensorflow as tf

def pad_to_shape(grid, target_shape=(30, 30), pad_value=0):
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    h, w = grid.shape
    H, W = target_shape
    if h > H or w > W:
        raise ValueError(f"Cannot pad grid of shape ({h}, {w}) into ({H}, {W}). Grid is too large.")
    padded = np.full((H, W), pad_value, dtype=grid.dtype)
    padded[:h, :w] = grid
    return padded


def match_grid_orientation(y_true, y_pred_logits):
    """
    Detects if model output is transposed compared to y_true and aligns it.

    Parameters:
        y_true: np.ndarray or tf.Tensor of shape (B, H, W) or (H, W)
        y_pred_logits: tf.Tensor of shape (B, H, W, C) or (B, W, H, C)

    Returns:
        tf.Tensor: aligned logits with shape matching y_true
    """
    if isinstance(y_true, tf.Tensor):
        y_true_shape = tf.shape(y_true)
        h_true, w_true = y_true_shape[-2], y_true_shape[-1]
    else:
        h_true, w_true = y_true.shape[-2:]

    h_pred, w_pred = y_pred_logits.shape[1], y_pred_logits.shape[2]

    # Heuristic: only transpose if H and W are obviously swapped and the difference is large enough
    if abs(h_true - h_pred) + abs(w_true - w_pred) > 0 and (h_true, w_true) == (w_pred, h_pred):
        return tf.transpose(y_pred_logits, perm=[0, 2, 1, 3])
    return y_pred_logits



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