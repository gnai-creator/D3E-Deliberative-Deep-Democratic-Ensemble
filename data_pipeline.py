# data_pipeline.py
def load_data_batches():
    from dummy_data_loader import load_dummy_test_data  # mock
    batches = []
    for block_index in range(5):  # ou len(X_test)
        X_test, raw_input, task_id = load_dummy_test_data(block_index)
        batches.append((X_test[0], raw_input, block_index, task_id))
    return batches
