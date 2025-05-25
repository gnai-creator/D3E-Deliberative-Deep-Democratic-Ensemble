# data_pipeline.py
import tensorflow as tf

def load_data_batches(challenges="", task_ids="", num_models=5):
    from data_loader import load_data  # mock
    batches = []
    for block_index in range(num_models):  # ou len(X_test)
        X_train, X_val, Y_train, Y_val, _, _, X_test, info_train, info_val, task_id, raw_input, raw_test_inputs = load_data(
            challenges=challenges,
            block_index=block_index,
            task_ids=task_ids,
            block_size=1,
            pad_value=0,         
            vocab_size=10,      
            model_idx=block_index
        )
        # Modelo 4 (index 4) precisa dos 40 canais (não quebrado em subcanais)
        if block_index == 4:
            x = tf.reshape(X_train[0], (30, 30, 40))  # ou o reshape apropriado
        else:
            x = X_train[0]
        batches.append((x, X_val, Y_train, Y_val, X_test[0], raw_input, block_index, task_id))
    return batches
