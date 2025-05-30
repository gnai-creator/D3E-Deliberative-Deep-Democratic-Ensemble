# data_pipeline.py
import tensorflow as tf
from data_loader import load_data

def load_data_batches(challenges="", task_ids="", num_models=5, model_idx=0, block_idx=0):
    batches = []
    # Corrigido para carregar apenas o bloco correspondente ao block_idx atual
    block_index = block_idx

    X_train, X_val, Y_train, Y_val, X_test, info_train, info_val, task_id, raw_input, raw_test_inputs = load_data(
        challenges=challenges,
        block_index=block_idx,
        task_ids=task_ids,
        block_size=1,
        pad_value=-1,
        vocab_size=10,
        model_idx=model_idx,
        block_idx=block_idx
    )

    batches.append((X_train, X_val, Y_train, Y_val, X_test, raw_input, block_index, task_id))
    return batches

