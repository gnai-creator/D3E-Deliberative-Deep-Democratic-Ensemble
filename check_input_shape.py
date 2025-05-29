import tensorflow as tf

def shape_guard(expected_shape=1, name="UNKNOWN"):
    """
    Decorator to validate input tensor shapes for model calls.
    
    expected_shape: Tuple or list of expected dimensions. Use None to skip checking that dim.
    name: Name of the model or role for logging clarity.
    """
    def decorator(func):
        def wrapper(self, x, *args, **kwargs):
            x_shape = tf.shape(x)
            static_shape = x.shape.as_list()

            # Check against expected shape
            if expected_shape is not None:
                mismatches = []
                for i, expected_dim in enumerate(expected_shape):
                    if expected_dim is not None and static_shape[i] != expected_dim:
                        mismatches.append((i, expected_dim, static_shape[i]))

                if mismatches:
                    msg = f"[SECURITY] Shape violation in '{name}' — received {static_shape}, expected {expected_shape}."
                    for idx, exp, got in mismatches:
                        msg += f"\n - Dimension {idx}: expected {exp}, got {got}"
                    raise ValueError(msg)

            print(f"[SECURITY] '{name}' input accepted with shape {static_shape}")
            return func(self, x, *args, **kwargs)
        return wrapper
    return decorator
