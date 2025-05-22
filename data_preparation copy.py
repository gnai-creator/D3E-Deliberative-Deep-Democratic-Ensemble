import tensorflow as tf
import numpy as np
from runtime_utils import log, pad_to_shape
from sklearn.model_selection import train_test_split
from data_augmentation import augment_dataset_rotation_flip

def get_dataset(block_index, task_ids, train_challenges, block_size, pad_value, vocab_size):
    start = block_index * block_size
    end = start + block_size
    block_task_ids = task_ids[start:end]

    X_train, y_train = [], []
    X_test_raw = []

    log(f"[INFO] Treinando bloco {block_index} com tasks: {block_task_ids}")

    for task_id in block_task_ids:
        task_data = train_challenges.get(task_id, {})

        for example in task_data.get("train", []):
            inp = np.array(example["input"])
            out = np.array(example["output"])
            X_train.append(inp)
            y_train.append(out)

        for test_example in task_data.get("test", []):
            inp = np.array(test_example["input"])
            X_test_raw.append(inp)

    if not X_train or not y_train:
        raise ValueError(f"[ERRO] Nenhum dado de treino encontrado no bloco {block_index}.")

    X_all_raw = [pad_to_shape(x, pad_value=pad_value) for x in X_train]
    y_all_raw = [pad_to_shape(y, pad_value=pad_value) for y in y_train]
    X_test_padded = [pad_to_shape(x, pad_value=pad_value) for x in X_test_raw]

    X_all_raw = tf.convert_to_tensor(X_all_raw, dtype=tf.int32)
    y_all = tf.convert_to_tensor(y_all_raw, dtype=tf.int32)
    X_test = tf.convert_to_tensor(X_test_padded, dtype=tf.int32)

    pad_mask = tf.cast(X_all_raw != pad_value, tf.float32)
    X_all_safe = tf.where(X_all_raw == pad_value, 0, X_all_raw)
    X_all = tf.one_hot(X_all_safe, depth=vocab_size, dtype=tf.float32) * tf.expand_dims(pad_mask, -1)

    sample_weight = tf.cast(y_all != pad_value, tf.float32)
    y_all_clean = tf.where(y_all == pad_value, 0, y_all)

    if len(X_all) < 2:
        log(f"[WARN] Apenas uma amostra no bloco {block_index}. Usando a mesma para treino e validação.")
        X_train_final = X_val_final = X_all
        y_train_final = y_val_final = y_all_clean
        sw_train = sw_val = sample_weight
        X_aug = X_train_final
        y_aug = y_train_final
    else:
        X_train_final, X_val_final, y_train_final, y_val_final, sw_train, sw_val = train_test_split(
            X_all.numpy(), y_all_clean.numpy(), sample_weight.numpy(), test_size=0.2, random_state=42
        )
        # Convert to tensors
        X_train_final = tf.convert_to_tensor(X_train_final, dtype=tf.float32)
        y_train_final = tf.convert_to_tensor(y_train_final, dtype=tf.int32)
        sw_train = tf.convert_to_tensor(sw_train, dtype=tf.float32)

        X_val_final = tf.convert_to_tensor(X_val_final, dtype=tf.float32)
        y_val_final = tf.convert_to_tensor(y_val_final, dtype=tf.int32)
        sw_val = tf.convert_to_tensor(sw_val, dtype=tf.float32)

        # Augmentation happens here
        X_aug, y_aug = augment_dataset_rotation_flip(X_train_final, y_train_final)


    # --- SHAPE FIXING ---
    X_aug = tf.expand_dims(X_aug, -1)                # (batch, 30, 30, 10, 1)
    X_aug = tf.tile(X_aug, [1, 1, 1, 1, 10])          # (batch, 30, 30, 10, 10)

    X_val_final = tf.expand_dims(X_val_final, -1)
    X_val_final = tf.tile(X_val_final, [1, 1, 1, 1, 10])

    # 🧼 Ensure sw_train and sw_val match y shapes
    if sw_train.shape[0] == 1:
        batch_size = tf.shape(y_aug)[0] if isinstance(y_aug, tf.Tensor) else len(y_aug)
        sw_train = tf.repeat(sw_train, repeats=batch_size, axis=0)
    if tf.shape(sw_val)[0] == 1:
        sw_val = tf.repeat(sw_val, repeats=tf.shape(y_val_final)[0], axis=0)


    log("Shapes retornados por get_dataset:")
    log(f"x_train_final: {X_train_final.shape}")
    log(f"y_train_final: {y_train_final.shape}")
    log(f"sw_train: {sw_train.shape}")


    return (
        X_aug,                                       # Tensor
        X_val_final,                                 # Tensor
        tf.convert_to_tensor(y_aug, dtype=tf.int32), # Tensor ← MAKE SURE THIS IS TENSOR
        y_val_final,                                 # Tensor
        sw_train,                                    # Tensor
        sw_val,                                      # Tensor
        X_test,
        X_all_raw
    )
