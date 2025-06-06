import tensorflow as tf
from model_compile import compile_model
from SimuV1 import SimuV1
from runtime_utils import log

# Configurações de cada modelo: classe e hidden_dim
MODEL_CONFIGS = {
    0: (SimuV1, 128),   # jurada
    1: (SimuV1, 128),   # Advogada
    2: (SimuV1, 128),   # Juiz
    3: (SimuV1, 128),   # Promotor

}

_model_cache = {}

def load_model(index, learning_rate):
    if index in _model_cache:
        return _model_cache[index]

    if index not in MODEL_CONFIGS:
        raise ValueError(f"[FATAL] Índice de modelo inválido: {index}")

    model_class, hidden_dim = MODEL_CONFIGS[index]
    model = model_class(hidden_dim=hidden_dim)
    model = compile_model(model, lr=learning_rate)

    dummy_shape = (1, 30, 30, 3, 1) 
    dummy_input = tf.zeros(dummy_shape, dtype=tf.float32)
    _ = model(dummy_input, training=False)
    log(f"[DEBUG] Criado modelo {index} com input dummy shape: {dummy_shape}")
    _model_cache.clear()
    _model_cache[index] = model
    return model
