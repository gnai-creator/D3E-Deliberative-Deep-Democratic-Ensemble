import tensorflow as tf
import numpy as np
from runtime_utils import log, pad_to_shape
from sklearn.model_selection import train_test_split
from data_augmentation import augment_with_class_replacement

def get_dataset(block_index, task_ids, train_challenges, block_size, pad_value, vocab_size):
    start = block_index * block_size
    end = start + block_size
    block_task_ids = task_ids[start:end]
    X_train, y_train = [], []
    X_test = []
    test_list = []

    # Aqui você colocaria o seu treino com block_train_challenges
    log(f"[INFO] Treinando bloco {block_index} com tasks: {block_task_ids}")

    for task_idx in range(block_size):
        train_list = train_challenges.get(block_task_ids[task_idx], [])
        for task_ids_keys in train_list:
            if task_ids_keys == "train":
                for task_input_idx in range(len(train_list[task_ids_keys])):
                    X_train.append(train_list[task_ids_keys][task_input_idx]['input'])
                    y_train.append(train_list[task_ids_keys][task_input_idx]['output'])
            else:
                test_list.append(train_list[task_ids_keys])

    for test_idx in range(len(test_list)):
        X_test.append(test_list[test_idx][0]['input'])

    # Padding
    X_all_raw = [pad_to_shape(np.array(x), pad_value=pad_value) for x in X_train]
    y_all = [pad_to_shape(np.array(y), pad_value=pad_value) for y in y_train]
    X_test = [pad_to_shape(np.array(x), pad_value=pad_value) for x in X_test]

    # Tensores
    X_all_raw = tf.convert_to_tensor(X_all_raw, dtype=tf.int32)
    y_all = tf.convert_to_tensor(y_all, dtype=tf.int32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.int32)

    # Criar máscara de padding
    pad_mask = tf.cast(X_all_raw != pad_value, tf.float32)

    # Substituir valores de padding por 0 para one-hot seguro
    X_all_safe = tf.where(X_all_raw == pad_value, 0, X_all_raw)
    X_all = tf.one_hot(X_all_safe, depth=vocab_size, dtype=tf.float32) * tf.expand_dims(pad_mask, -1)

    # Processar o y_all
    sample_weight = tf.cast(y_all != pad_value, tf.float32)
    y_all = tf.where(y_all == pad_value, 0, y_all)

    # Split
    X_train_final, X_val_final, y_train_final, y_val_final, sw_train, sw_val = train_test_split(
        X_all.numpy(), y_all.numpy(), sample_weight.numpy(), test_size=0.2, random_state=42
    )

    return (
        tf.convert_to_tensor(X_train_final, dtype=tf.float32),
        tf.convert_to_tensor(X_val_final, dtype=tf.float32),
        tf.convert_to_tensor(y_train_final, dtype=tf.int32),
        tf.convert_to_tensor(y_val_final, dtype=tf.int32),
        tf.convert_to_tensor(sw_train, dtype=tf.float32),
        tf.convert_to_tensor(sw_val, dtype=tf.float32),
        X_test,
        X_all_raw
    )
