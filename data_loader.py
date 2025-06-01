import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from runtime_utils import log
from metrics import expand_grid_to_30x30x1, add_judge_channel, expand_grid_to_30x30x1_Y

def load_data(block_index, task_ids, challenges, block_size, pad_value, vocab_size, block_idx, model_idx):
    start_idx = block_index * block_size
    end_idx = start_idx + block_size
    block_task_ids = task_ids[start_idx:end_idx]
    log(f"[CHECK] block_index={block_index}, task_ids[{start_idx}:{end_idx}] = {block_task_ids}")

    raw_inputs = []
    raw_test_inputs = []
    X = []
    Y = []
    X_test = []
    info = []

    for task_id in block_task_ids:
        if task_id not in challenges:
            log(f"[ERRO] task_id '{task_id}' não encontrado em challenges. Pulando.")
            continue
        log(f"Treinando task_id: {task_id}")
        challenge = challenges[task_id]
        try:
            input_grid = np.array(challenge["train"][0]["input"], dtype=np.int32)
            output_grid = np.array(challenge["train"][0]["output"], dtype=np.int32)
            test_input_grid = np.array(challenge["test"][0]["input"], dtype=np.int32)
            raw_inputs.append(input_grid)
            raw_test_inputs.append(test_input_grid)
        except Exception as e:
            log(f"[BROKE]: {e}")
            continue

        max_h = max(input_grid.shape[0], output_grid.shape[0], test_input_grid.shape[0])
        max_w = max(input_grid.shape[1], output_grid.shape[1], test_input_grid.shape[1])
        if max_h > 30 or max_w > 30:
            log(f"[WARN] Grid maior que 30x30: {max_h}x{max_w} — pulando")
            continue

        # Converte para (30, 30, 1)
        input_grid = expand_grid_to_30x30x1(input_grid, pad_value)
        output_grid = expand_grid_to_30x30x1_Y(output_grid, pad_value)
        test_input_grid = expand_grid_to_30x30x1(test_input_grid, pad_value)

        # Adiciona canal de juízo após expansão
        input_grid = add_judge_channel(input_grid, juizo_value=0, confidence_value=1)
        output_grid = add_judge_channel(output_grid, juizo_value=1, confidence_value=1)
        test_input_grid = add_judge_channel(test_input_grid, juizo_value=0, confidence_value=1)

        # output_grid = output_grid / 9.0  # Normaliza para range [0, 1]

        X.append(input_grid)
        Y.append(output_grid)
        X_test.append(test_input_grid)

        info.append({"task_id": task_id})

    X = np.array(X)
    Y = np.array(Y)
    X_test = np.array(X_test)

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

    # sw_train = np.ones_like(Y_train[..., 0], dtype=np.float32)
    # sw_val = np.ones_like(Y_val[..., 0], dtype=np.float32)
    # log(f"[DEBUG] X_TRAIN SHAPE FINAL : {X_train.shape}")
    # log(f"[DEBUG] X_VAL SHAPE FINAL : {X_val.shape}")
    # log(f"[DEBUG] Y_TRAIN SHAPE FINAL : {Y_train.shape}")
    # log(f"[DEBUG] Y_VAL SHAPE FINAL : {Y_val.shape}")
    # log(f"[DEBUG] X_TESTE SHAPE FINAL : {X_test.shape}")

    return (
        tf.convert_to_tensor(X_train, dtype=tf.float32),
        tf.convert_to_tensor(X_val, dtype=tf.float32),
        tf.convert_to_tensor(Y_train, dtype=tf.float32),
        tf.convert_to_tensor(Y_val, dtype=tf.float32),
        # tf.convert_to_tensor(sw_train, dtype=tf.float32),
        # tf.convert_to_tensor(sw_val, dtype=tf.float32),
        tf.convert_to_tensor(X_test, dtype=tf.float32),
        info_train,
        info_val,
        task_id,
        raw_inputs,
        raw_test_inputs
    )
