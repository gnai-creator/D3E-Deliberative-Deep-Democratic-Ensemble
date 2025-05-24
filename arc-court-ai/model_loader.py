from models.SimuV1 import SimuV1
from models.SimuV2 import SimuV2
from models.SimuV3 import SimuV3
from models.SimuV4 import SimuV4
from models.SimuV5 import SimuV5
from model_compile import compile_model


def load_model(index):
             
    if index == 0:
        model = SimuV1(hidden_dim=128)
        model = compile_model(model, lr=LEARNING_RATE)
    elif index == 1:
        model = SimuV2(hidden_dim=256)
        model = compile_model(model, lr=LEARNING_RATE)
    elif index == 2:
        model = SimuV3(hidden_dim=128)
        model = compile_model(model, lr=LEARNING_RATE)
    elif index == 3:
        model = SimuV4(hidden_dim=256)
        model = compile_model(model, lr=LEARNING_RATE)
    else:
        model = SimuV5(hidden_dim=128)
        model = compile_model(model, lr=LEARNING_RATE)
                