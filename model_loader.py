from SimuV1 import SimuV1
from SimuV2 import SimuV2
from SimuV3 import SimuV3
from SimuV4 import SimuV4
from SimuV5 import SimuV5
from model_compile import compile_model
import tensorflow as tf

def load_model(index, learning_rate):
             
    if index == 0:
        model = SimuV1(hidden_dim=128)
        model = compile_model(model, lr=learning_rate)
    elif index == 1:
        model = SimuV2(hidden_dim=256)
        model = compile_model(model, lr=learning_rate)
    elif index == 2:
        model = SimuV3(hidden_dim=128)
        model = compile_model(model, lr=learning_rate)
    elif index == 3:
        model = SimuV4(hidden_dim=256)
        model = compile_model(model, lr=learning_rate)
    else:
        model = SimuV5(hidden_dim=128)
        model = compile_model(model, lr=learning_rate)

    dummy_input = tf.random.uniform((1, 16, 16, 1, 4))  # ajuste esse shape se necessário
    model(dummy_input, training=False)
    return model
                