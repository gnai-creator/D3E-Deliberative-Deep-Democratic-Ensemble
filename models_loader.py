import tensorflow as tf
from model_compile import compile_model
from SimuV1 import SimuV1


# Configurações de cada modelo: classe e hidden_dim
MODEL_CONFIGS = {
    0: (SimuV1, 128),   # jurada
    1: (SimuV1, 256),   # jurada
    2: (SimuV1, 128),   # jurada
    3: (SimuV1, 256),   # advogada
    4: (SimuV1, 128),   # juiza
    5: (SimuV1, 256),   # suprema
    6: (SimuV1, 256)    # promotor
}

def load_model(index, learning_rate):
    if index not in MODEL_CONFIGS:
        raise ValueError(f"[FATAL] Índice de modelo inválido: {index}")

    model_class, hidden_dim = MODEL_CONFIGS[index]
    model = model_class(hidden_dim=hidden_dim)
    model = compile_model(model, lr=learning_rate)

    # Chamada inicial com dummy_input para forçar construção
    dummy_shape = (1, 30, 30, 10, 40) if index >= 4 else (1, 30, 30, 10, 4)
    dummy_input = tf.zeros(dummy_shape, dtype=tf.float32)
    _ = model(dummy_input, training=False)

    return model
