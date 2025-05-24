from models.SimuV1 import SimuV1
from models.SimuV2 import SimuV2
from models.SimuV3 import SimuV3
from models.SimuV4 import SimuV4
from models.SimuV5 import SimuV5
from model_compile import compile_model


def load_model(index, learning_rate=0.001):

    model_classes = [SimuV1, SimuV2, SimuV3, SimuV4, SimuV5]

    if 0 <= index < len(model_classes):
        model = compile_model(model_classes[index](), lr=learning_rate)
        return model
    else:
        return None
