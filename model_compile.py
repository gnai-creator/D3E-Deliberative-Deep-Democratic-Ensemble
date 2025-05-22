import os
import tensorflow as tf
from runtime_utils import log
from shape_locator_net import ShapeLocatorNet, compile_shape_locator  # nome correto do arquivo

def model_compilation(index, learning_rate, vocab_size, block_index, result_dir):
    base_model = ShapeLocatorNet()

    inputs = tf.keras.Input(shape=(30, 30, 10, vocab_size))
    outputs = base_model(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model_dir = os.path.join(result_dir, "Model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{index}")
    weights_file = model_path + "_weights.keras"

    if os.path.exists(weights_file):
        try:
            model.load_weights(weights_file)
        except Exception as e:
            log(f"[WARN] Falha ao carregar pesos de {weights_file}: {e}")

    compile_shape_locator(model, lr=learning_rate)

    log(f"Modelo {index} compilado (ShapeLocatorNet)!")
    return model, model_path, base_model
