import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from runtime_utils import log
from metrics import standardize_grid_shapes, pad_to_30x30_top_left, pad_to_30x30_top_left_single, add_judge_channel



def load_data(block_index, task_ids, challenges, block_size, pad_value, vocab_size, block_idx, model_idx):
    start_idx = block_index * block_size
    end_idx = start_idx + block_size
    block_task_ids = task_ids[start_idx:end_idx]

    raw_inputs = []
    raw_test_inputs = []
    X = []
    Y = []
    X_test = []
    info = []

    for task_id in block_task_ids:
        log(f"Treinando task_id: {task_id}")
        challenge = challenges[task_id]
        try:
            input_grid = np.array(challenge["train"][0]["input"], dtype=np.int32)
            output_grid = np.array(challenge["train"][0]["output"], dtype=np.int32)
            test_input_grid = np.array(challenge["test"][0]["input"], dtype=np.int32)
            raw_inputs.append(input_grid)
            raw_test_inputs.append(test_input_grid)

            log(f"TRAIN TASK {task_id}  SHAPE: - {input_grid.shape}")
            log(f"TEST TASK {task_id}  SHAPE: - {test_input_grid.shape}")
        except Exception as e:
            log(f"[BROKE]: {e}")
            continue

        max_h = max(input_grid.shape[0], output_grid.shape[0], test_input_grid.shape[0])
        max_w = max(input_grid.shape[1], output_grid.shape[1], test_input_grid.shape[1])
        if max_h > 30 or max_w > 30:
            log(f"[WARN] Grid maior que 30x30: {max_h}x{max_w} — pulando")
            continue
        
        if model_idx >= 4:
            X.append(add_judge_channel(input_grid, juizo_value=0, channel_value=10, confidence_value=40))
            X_test.append(add_judge_channel(test_input_grid, juizo_value=0, channel_value=10, confidence_value=40))
            Y.append(add_judge_channel(output_grid, juizo_value=1, channel_value=10, confidence_value=40))
        else:
            X.append(add_judge_channel(input_grid, juizo_value=0, channel_value=10, confidence_value=4))
            X_test.append(add_judge_channel(test_input_grid, juizo_value=0, channel_value=10, confidence_value=4))
            Y.append(add_judge_channel(output_grid, juizo_value=1, channel_value=10, confidence_value=4))

        info.append({"task_id": task_id})

    X, Y = standardize_grid_shapes(X, Y)
    X, Y = pad_to_30x30_top_left(X, Y)
    X_test, _ = standardize_grid_shapes(X_test, Y)
    X_test = pad_to_30x30_top_left_single(X=X_test)

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

    if len(X) > 1:
        X_train, X_val, Y_train, Y_val, info_train, info_val = train_test_split(
            X, Y, info, test_size=0.2, random_state=42
        )
    else:
        X_train, X_val = X, X
        Y_train, Y_val = Y, Y
        info_train, info_val = info, info

    if len(X_train.shape) == 4:
        X_train = tf.expand_dims(X_train, axis=0)
    
   
    # if len(Y_train.shape) == 4:
    #     Y_train = tf.expand_dims(Y_train, axis=0)
    # if len(Y_val.shape) == 4:
    #     Y_val = tf.expand_dims(Y_val, axis=0)
    log(f"[DEBUG] X_TRAIN SHAPE : {X_train.shape}")
    log(f"[DEBUG] Y_VAL SHAPE : {Y_val.shape}")
    log(f"[DEBUG] X_TEST SHAPE : {X_test.shape}")
    
    sw_train = np.ones_like(Y_train[..., 0], dtype=np.float32)
    sw_val = np.ones_like(Y_val[..., 0], dtype=np.float32)

    return (
        tf.convert_to_tensor(X_train, dtype=tf.float32),
        tf.convert_to_tensor(X_val, dtype=tf.float32),
        tf.cast(Y_train[:, :, :, 0, 0], dtype=tf.int32),
        tf.cast(Y_val[:, :, :, 0, 0], dtype=tf.int32),

        tf.convert_to_tensor(sw_train, dtype=tf.float32),
        tf.convert_to_tensor(sw_val, dtype=tf.float32),
        tf.convert_to_tensor(X_test, dtype=tf.float32),
        info_train,
        info_val,
        task_id,
        raw_inputs,
        raw_test_inputs
    )

