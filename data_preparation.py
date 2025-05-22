import tensorflow as tf
import numpy as np
from runtime_utils import log, pad_to_shape
from sklearn.model_selection import train_test_split

def get_dataset(block_index, task_ids, train_challenges, block_size, pad_value, vocab_size):
    start = block_index * block_size
    end = start + block_size
    block_task_ids = task_ids[start:end]

    X_train, y_train = [], []
    X_test_raw = []

    log(f"[INFO] Treinando bloco {block_index} com tasks: {block_task_ids}")

    # Coleta os dados de treino/teste do bloco
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

    # Padding das entradas e saídas
    X_all_raw = [pad_to_shape(x, pad_value=pad_value) for x in X_train]
    y_all_raw = [pad_to_shape(y, pad_value=pad_value) for y in y_train]
    X_test_padded = [pad_to_shape(x, pad_value=pad_value) for x in X_test_raw]

    # Conversão para tensor
    X_all_raw = tf.convert_to_tensor(X_all_raw, dtype=tf.int32)
    y_all = tf.convert_to_tensor(y_all_raw, dtype=tf.int32)
    X_test = tf.convert_to_tensor(X_test_padded, dtype=tf.int32)

    # Máscara de padding
    pad_mask = tf.cast(X_all_raw != pad_value, tf.float32)

    # Substituição segura do padding (por 0) para usar no one-hot
    X_all_safe = tf.where(X_all_raw == pad_value, 0, X_all_raw)
    X_all = tf.one_hot(X_all_safe, depth=vocab_size, dtype=tf.float32) * tf.expand_dims(pad_mask, -1)

    # Sample weights e limpeza do y
    sample_weight = tf.cast(y_all != pad_value, tf.float32)
    y_all_clean = tf.where(y_all == pad_value, 0, y_all)

    # Verifica se há amostras suficientes para split
    if len(X_all) < 2:
        # Não dá pra fazer train/val split com 1 exemplo só. Então copia tudo como treino e validação.
        log(f"[WARN] Apenas uma amostra no bloco {block_index}. Usando a mesma para treino e validação.")
        X_train_final = X_val_final = X_all
        y_train_final = y_val_final = y_all_clean
        sw_train = sw_val = sample_weight
    else:
        X_train_final, X_val_final, y_train_final, y_val_final, sw_train, sw_val = train_test_split(
            X_all.numpy(), y_all_clean.numpy(), sample_weight.numpy(), test_size=0.2, random_state=42
        )

    # Retorno como tensores
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
