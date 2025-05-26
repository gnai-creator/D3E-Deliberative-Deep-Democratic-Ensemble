import os
import numpy as np
import tensorflow as tf
from confidence_system import ConfidenceManager, avaliar_consenso_com_confiança
from metrics_utils import salvar_voto_visual
from runtime_utils import log


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

        # Advogada aprende com consenso dos jurados (0, 1, 2)
        juradas_preds = [tf.argmax(votos_models[f"modelo_{i}"], axis=-1) for i in range(3)]  # (1, 30, 30)
        stack = tf.stack(juradas_preds, axis=0)  # (3, 1, 30, 30)

        # Cálculo da moda por pixel (consenso estatístico)
        # Transforma para (1, 30, 30, 3) e calcula moda por pixel
        stack_transposed = tf.transpose(stack, [1, 2, 3, 0])  # (1, 30, 30, 3)
        flat = tf.reshape(stack_transposed, (-1, 3))  # (900, 3)

        def pixel_mode(x):
            with tf.device("/CPU:0"):
                bincount = tf.math.bincount(tf.cast(x, tf.int32), minlength=10)

            return tf.argmax(bincount)


        moda_flat = tf.map_fn(pixel_mode, flat, fn_output_signature=tf.int64)  # (900,)
        y_treino = tf.reshape(moda_flat, (1, 30, 30))  # (1, 30, 30)

        entrada_crua_adv = tf.expand_dims(entrada_crua, axis=-1)  # (1, 30, 30, 1)
        entrada_crua_adv = tf.expand_dims(entrada_crua_adv, axis=-1)  # (1, 30, 30, 1, 1)
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
        votos_visuais = []
        for i, v in votos_models.items():
            try:
                if len(v.shape) > 3 and v.shape[-1] > 1:
                    v = tf.argmax(v, axis=-1)  # pega a classe dominante
                v = tf.squeeze(v)
                votos_visuais.append(v)
            except Exception as e:
                log(f"[VISUAL] Erro ao preparar voto do modelo {i}: {e}")


        input_visual = input_tensor_outros[..., 0, 0]  # (1, 30, 30)
        input_visual = tf.squeeze(input_visual)        # (30, 30)
        salvar_voto_visual(votos_visuais, iter_count, block_idx, input_visual, task_id=task_id, idx=idx)
        salvar_voto_visual(votos_visuais, idx, block_idx, input_visual, task_id=task_id, idx=idx)

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
