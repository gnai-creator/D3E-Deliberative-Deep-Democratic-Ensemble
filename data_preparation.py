from runtime_utils import log

def get_dataset(block_index, block_size, task_ids, train_challenges):
    start = block_index * block_size
    end = start + block_size
    block_task_ids = task_ids[start:end]
    block_train_challenges = {k: train_challenges[k] for k in block_task_ids}
    X_train, y_train = [],[]
    X_test, y_test = [], []
    test_list = []
    input_list = []

    # Aqui você colocaria o seu treino com block_train_challenges
    log(f"[INFO] Treinando bloco {block_index} com tasks: {block_task_ids}")

    for task_idx in range(block_size):
        # log(f"task_idx: {task_idx}")
        # log(f"Block_tasks_ids: {block_task_ids[task_idx]}")
    
        train_list = train_challenges.get(block_task_ids[task_idx], [])
        # log(f"train_list: {train_list}")
        for task_ids_keys in train_list:
            # log(task_ids_keys)
            if task_ids_keys == "train":
                for task_input_idx in range(len(train_list[task_ids_keys])):
                        X_train.append(train_list[task_ids_keys][task_input_idx]['input'])
                        y_train.append(train_list[task_ids_keys][task_input_idx]['output'])
                    
            else:                    
                test_list.append(train_list[task_ids_keys])
    
    for test_idx in range(len(test_list)):
        # log(f"test_list[test_idx][0]['input'] : {test_list[test_idx][0]['input']}")
        X_test.append(test_list[test_idx][0]['input'])
    
    return X_train, y_train, X_test