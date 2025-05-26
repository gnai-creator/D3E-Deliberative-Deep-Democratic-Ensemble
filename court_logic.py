import os
import numpy as np
import tensorflow as tf
from confidence_system import ConfidenceManager, avaliar_consenso_com_confiança
from metrics_utils import salvar_voto_visual
from runtime_utils import log

def arc_court_supreme(models, input_tensor_outros, task_id=None, block_idx=None,
                      max_cycles=150, tol=0.98, epochs=1, confidence_threshold=0.5,
                      confidence_manager=[]):
    log(f"[SUPREMA] Iniciando deliberação para o bloco {block_idx} — task {task_id}")
    manager = confidence_manager
    votos_models = {}
    iter_count = 0

    while iter_count < max_cycles:
        log(f"[CICLO] Iteração {iter_count}")
        votos_models.clear()

        juradas = models[:3]
        advogada = models[3]
        juiza = models[4]
        suprema_juiza = models[5]

        entrada_crua = input_tensor_outros[..., 0, 0]  # (1, H, W)

        # Juradas votam
        for i, modelo in enumerate(juradas):
            nome = f"modelo_{i}"
            voto = modelo(input_tensor_outros, training=False)
            votos_models[nome] = voto

        # Advogada aprende com Jurada 0
        y_treino = (tf.argmax(votos_models["modelo_0"], axis=-1) +
                    tf.random.uniform(shape=(1, 30, 30), maxval=3, dtype=tf.int64)) % 10

        entrada_crua_adv = tf.reshape(entrada_crua, [1, 30, 30, 1, 1])
        entrada_crua_adv = pad_or_truncate_channels(entrada_crua_adv, 4)
        advogada.fit(entrada_crua_adv, y_treino, epochs=epochs, verbose=0)
        votos_models["modelo_3"] = advogada(input_tensor_outros, training=False)

        # Juíza aprende com advogada
        y_juiza = tf.argmax(votos_models["modelo_3"], axis=-1)
        entrada_juiza = entrada_crua  # shape: (1, 30, 30)
        entrada_juiza = tf.expand_dims(entrada_juiza, axis=-1)  # → (1, 30, 30, 1)
        entrada_juiza = tf.expand_dims(entrada_juiza, axis=-1)  # → (1, 30, 30, 1, 1)
        entrada_juiza = pad_or_truncate_channels(entrada_juiza, 40)  # → (1, 30, 30, 1, 40)
        votos_models["modelo_4"] = juiza(entrada_juiza, training=False)

        # Suprema Juíza aprende com dado cru (entrada original)
        entrada_crua_suprema = tf.expand_dims(entrada_crua, axis=-1)  # (1, 30, 30, 1)
        entrada_crua_suprema = tf.expand_dims(entrada_crua_suprema, axis=-1)  # (1, 30, 30, 1, 1)
        entrada_crua_suprema = pad_or_truncate_channels(entrada_crua_suprema, 40)  # (1, 30, 30, 1, 40)

        y_suprema = tf.argmax(votos_models["modelo_4"], axis=-1)
        suprema_juiza.fit(entrada_crua_suprema, y_suprema, epochs=epochs, verbose=0)
        votos_models["modelo_5"] = suprema_juiza(entrada_crua_suprema, training=False)

        # Visual
        salvar_voto_visual(list(votos_models.values()), iter_count, block_idx, input_tensor_outros, task_id=task_id)

        consenso = avaliar_consenso_com_confiança(
            votos_models, confidence_manager=manager,
            required_votes=5, confidence_threshold=confidence_threshold
        )

        log(f"[CONSENSO] Score de consenso = {consenso:.3f} (limite = {tol})")

        if consenso >= tol:
            y_eval = tf.argmax(votos_models["modelo_4"], axis=-1)
            loss, acc = suprema_juiza.evaluate(entrada_crua_suprema, y_eval, verbose=0)
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

    # return votos_models["modelo_5"]


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
