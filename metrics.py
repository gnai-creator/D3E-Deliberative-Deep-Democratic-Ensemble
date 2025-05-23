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
    tf.debugging.assert_rank(y_pred_logits, 4, message="Expected y_pred_logits to be rank 4")
    tf.debugging.assert_rank(y_true, 3, message="Expected y_true to be rank 3")

    shape_pred = tf.shape(y_pred_logits)
    shape_true = tf.shape(y_true)

    h_pred, w_pred = shape_pred[1], shape_pred[2]
    h_true, w_true = shape_true[1], shape_true[2]

    should_transpose = tf.logical_and(
        tf.not_equal(h_true, h_pred),
        tf.logical_and(
            tf.equal(h_true, w_pred),
            tf.equal(w_true, h_pred)
        )
    )

    y_pred_aligned = tf.cond(
        should_transpose,
        lambda: tf.transpose(y_pred_logits, perm=[0, 2, 1, 3]),
        lambda: y_pred_logits
    )

    return y_pred_aligned





def pad_to_shape_batch(grids, target_shape, pad_value=0):
    if isinstance(grids, np.ndarray) and grids.ndim == 2:
        grids = [grids]  # wrap single example as list
    padded_batch = [pad_to_shape(grid, target_shape, pad_value) for grid in grids]
    return np.stack(padded_batch)

def align_shapes(a, b, pad_value=0):
    H = max(a.shape[0], b.shape[0])
    W = max(a.shape[1], b.shape[1])
    def pad_to(x, H, W):
        padded = np.full((H, W), pad_value, dtype=x.dtype)
        padded[:x.shape[0], :x.shape[1]] = x
        return padded
    return pad_to(a, H, W), pad_to(b, H, W)


class OrientationAwareSparseCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = match_grid_orientation(y_true, y_pred)
        return super().update_state(y_true, y_pred, sample_weight)

class ShapeAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="shape_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self._accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._accuracy.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self._accuracy.result()

    def reset_states(self):
        self._accuracy.reset_states()

    
    def variables(self):
        return self._accuracy._variables  # ou use _checkpoint_dependencies para versões mais novas

    def trainable_variables(self):
        return self._accuracy._trainable_variables










