import tensorflow as tf
from model_compile import compile_model
from SimuV1 import SimuV1
from SimuV2 import SimuV2
from SimuV3 import SimuV3
from SimuV4 import SimuV4
from SimuV5 import SimuV5

MODEL_CONFIGS = {
    0: (SimuV1, 128),
    1: (SimuV2, 256),
    2: (SimuV3, 128),
    3: (SimuV4, 256),
    4: (SimuV5, 128),
}

DUMMY_INPUT_SHAPE = (1, 30, 30, 1, 2)  # atualize se necessário

def load_model(index, learning_rate):
    if index not in MODEL_CONFIGS:
        raise ValueError(f"[FATAL] Índice de modelo inválido: {index}")

    model_class, hidden_dim = MODEL_CONFIGS[index]
    model = model_class(hidden_dim=hidden_dim)

    # Compila antes de chamar o modelo (seguindo TF 2.x compatibilidade)
    model = compile_model(model, lr=learning_rate)

    # Força criação de pesos para todos os submodelos
    dummy_input = tf.random.uniform(DUMMY_INPUT_SHAPE)
    try:
        _ = model(dummy_input, training=False)
    except Exception as e:
        raise RuntimeError(f"[FATAL] Falha ao ativar modelo {index}: {e}")

    return model
