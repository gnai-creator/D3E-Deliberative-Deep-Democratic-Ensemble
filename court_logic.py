import os
import numpy as np
import tensorflow as tf
from confidence_system import avaliar_consenso_com_confiança
from metrics_utils import salvar_voto_visual
from runtime_utils import log


def pixelwise_mode(stack):
    stack = tf.transpose(stack, [1, 2, 3, 0])  # (1, 30, 30, N)
    flat = tf.reshape(stack, (-1, stack.shape[-1]))  # (900, N)

    def pixel_mode(x):
        with tf.device("/CPU:0"):
            bincount = tf.math.bincount(tf.cast(x, tf.int32), minlength=10)
        return tf.argmax(bincount)

    moda_flat = tf.map_fn(pixel_mode, flat, fn_output_signature=tf.int64)
    return tf.reshape(moda_flat, (1, 30, 30))


def pad_or_truncate_channels(tensor, target_channels=40):
    current_channels = tensor.shape[-1]
    rank = len(tensor.shape)

    if current_channels == target_channels:
        return tensor
    elif current_channels < target_channels:
        padding = target_channels - current_channels
        paddings = [[0, 0]] * rank
        paddings[-1] = [0, padding]
        return tf.pad(tensor, paddings)
    else:
        return tensor[..., :target_channels]


def prepare_input_for_model(model_index, base_input):
    if model_index in [0, 1, 2, 3]:  # Juradas e Advogada
        return pad_or_truncate_channels(base_input, 4)
    else:  # Juíza e Suprema
        return pad_or_truncate_channels(base_input, 40)



def arc_court_supreme(models, input_tensor_outros, task_id=None, block_idx=None,
                      max_cycles=10, tol=0.98, epochs=1, confidence_threshold=0.5,
                      confidence_manager=[], idx=0):
    log(f"[SUPREMA] Iniciando deliberação para o bloco {block_idx} — task {task_id}")
    manager = confidence_manager
    votos_models = {}
    iter_count = 0

    while iter_count < max_cycles:
        log(f"[CICLO] Iteração {iter_count}")
        votos_models.clear()

        for i, modelo in enumerate(models):
            input_model = prepare_input_for_model(i, input_tensor_outros)
            votos_models[f"modelo_{i}"] = modelo(input_model, training=False)

        # Aprendizado
        # Juradas aprendem com advogada
        y_juradas = tf.argmax(votos_models["modelo_3"], axis=-1)
        for i in range(3):
            models[i].fit(prepare_input_for_model(i, input_tensor_outros), y_juradas, epochs=epochs, verbose=0)

        # Advogada aprende com suprema
        y_advogada = tf.argmax(votos_models["modelo_5"], axis=-1)
        models[3].fit(prepare_input_for_model(3, input_tensor_outros), y_advogada, epochs=epochs, verbose=0)

        # Juíza aprende com juradas + advogada
        stack_juiza = [tf.argmax(votos_models[f"modelo_{i}"], axis=-1) for i in range(4)]
        y_juiza = pixelwise_mode(tf.stack(stack_juiza, axis=0))
        models[4].fit(prepare_input_for_model(4, input_tensor_outros), y_juiza, epochs=epochs, verbose=0)

        # Suprema aprende com todos os anteriores
        stack_suprema = [tf.argmax(votos_models[f"modelo_{i}"], axis=-1) for i in range(5)]
        y_suprema = pixelwise_mode(tf.stack(stack_suprema, axis=0))
        models[5].fit(prepare_input_for_model(5, input_tensor_outros), y_suprema, epochs=epochs, verbose=0)

        # Atualiza votos após treinamento
        for i, modelo in enumerate(models):
            input_model = prepare_input_for_model(i, input_tensor_outros)
            votos_models[f"modelo_{i}"] = modelo(input_model, training=False)

        # Visual
        votos_visuais = []
        for i, v in votos_models.items():
            try:
                if len(v.shape) > 3 and v.shape[-1] > 1:
                    v = tf.argmax(v, axis=-1)
                v = tf.squeeze(v)
                votos_visuais.append(v)
            except Exception as e:
                log(f"[VISUAL] Erro ao preparar voto do modelo {i}: {e}")

        input_visual = tf.squeeze(input_tensor_outros[..., 0, 0])
        salvar_voto_visual(votos_visuais, iter_count, block_idx, input_visual, task_id=task_id, idx=idx)

        # Consenso
        consenso = avaliar_consenso_com_confiança(
            votos_models, confidence_manager=manager,
            required_votes=5, confidence_threshold=confidence_threshold
        )
        log(f"[CONSENSO] Score de consenso = {consenso:.3f} (limite = {tol})")

        if consenso >= tol:
            y_eval = tf.argmax(votos_models["modelo_4"], axis=-1)
            loss, acc = models[5].evaluate(prepare_input_for_model(5, input_tensor_outros), y_eval, verbose=0)
            log(f"[SUPREMA] Avaliação: acc = {acc:.3f}, loss = {loss:.6f}")

            if acc < 1.0 or loss > 0.001:
                log("[SUPREMA] Não aceita o consenso — requisita nova deliberação")
                iter_count += 1
                continue
            else:
                log("[SUPREMA] Confiança plena atingida — consenso final aceito")
                break

        iter_count += 1

    return {
        "class_logits": votos_models["modelo_5"],
        "consenso": consenso
    }
