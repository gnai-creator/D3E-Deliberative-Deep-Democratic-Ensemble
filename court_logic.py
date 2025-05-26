

import os
import numpy as np
import tensorflow as tf
from confidence_system import avaliar_consenso_com_confiança
from metrics_utils import salvar_voto_visual, preparar_voto_para_visualizacao
from runtime_utils import log
from metrics_utils import safe_squeeze
 
def normalizar_y_para_sparse(y):
    log(f"[DEBUG] Y SHAPE: {y.shape}")
    if y.shape.rank == 4 and y.shape[-1] != 1:
        y = tf.argmax(y, axis=-1)
        y = tf.expand_dims(y, axis=-1)
    return y

def pixelwise_mode(stack):
    stack = tf.transpose(stack, [1, 2, 3, 0])  # (1, 30, 30, N)
    flat = tf.reshape(stack, (-1, stack.shape[-1]))

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
    if model_index in [0, 1, 2, 3]:
        return pad_or_truncate_channels(base_input, 4)
    else:
        return pad_or_truncate_channels(base_input, 40)



def gerar_visualizacao_votos(votos_models, input_tensor_outros, idx, block_idx, task_id):
    votos_visuais = []
    try:
        for v in votos_models.values():
            resultado = preparar_voto_para_visualizacao(v)
            if resultado is not None:
                votos_visuais.append(resultado)
    except Exception as e:
        log(f"[VISUAL] Erro ao processar votos_models: {e}")


    try:
        # Extrai canal 0 e feature 0 corretamente
        v = input_tensor_outros
        if v.shape[-1] >= 1 and v.shape[-2] >= 1:
            input_visual = v[0, :, :, 0, 0]  # shape: [30, 30]
        else:
            input_visual = tf.zeros((30, 30), dtype=tf.int32)
        if len(input_visual.shape) != 2:
            raise ValueError
    except:
        log("[VISUAL] input_visual com shape inesperado, substituindo por zeros.")
        input_visual = tf.zeros((30, 30), dtype=tf.int32)
    salvar_voto_visual(votos_visuais, idx, block_idx, input_visual, task_id=task_id, idx=idx)


def arc_court_supreme(models, input_tensor_outros, task_id=None, block_idx=None,
                      max_cycles=10, tol=0.98, epochs=60, confidence_threshold=0.5,
                      confidence_manager=[], idx=0):
    log(f"[SUPREMA] Iniciando deliberação para o bloco {block_idx} — task {task_id}")
    manager = confidence_manager
    votos_models = {}
    iter_count = 0

    while iter_count < max_cycles:
        log(f"[CICLO] Iteração {iter_count}")
        votos_models.clear()
        log("[CORTE]Iniciando tribunal. Modelos irão fazer uma previsão do input cru")
        if iter_count == 0:
            for i in range(6):
                x_i = prepare_input_for_model(i, input_tensor_outros)
                votos_models[f"modelo_{i}"] = models[i](x_i, training=False)

        y_juradas = tf.argmax(votos_models["modelo_3"], axis=-1)
        y_juradas = tf.cast(tf.expand_dims(y_juradas, axis=-1), dtype=tf.int64)
        log(f"[DEBUG] y_juradas shape: {y_juradas.shape}")

        for i in range(3):
            x_i = prepare_input_for_model(i, input_tensor_outros)

            log(f"[JURADO {i}] treina com input")
            if i == 0:
                drop_mask = tf.random.stateless_uniform((1, 30, 30), seed=[42, iter_count]) > 0.95
                noise = tf.random.stateless_uniform((1, 30, 30), minval=1, maxval=3, dtype=tf.int64, seed=[43, iter_count])
                y_base = safe_squeeze(y_juradas, axis=-1)  # [1, 30, 30]
                y_ruidoso = tf.where(drop_mask, (y_base + noise) % 10, y_base)
                y_ruidoso = tf.expand_dims(y_ruidoso, axis=-1)  # [1, 30, 30, 1]

                # Corrige y caso esteja com 4D e último eixo diferente de 1
                if y_ruidoso.shape.rank == 4 and y_ruidoso.shape[-1] != 1:
                    y_ruidoso = safe_squeeze(y_ruidoso, axis=-1)
                assert y_ruidoso.shape[-1] == 1, f"y_ruidoso tem shape inválido: {y_ruidoso.shape}"
                models[i].fit(x_i, normalizar_y_para_sparse(y_ruidoso), epochs=epochs, verbose=0)

            elif i == 2 and iter_count > 0:
                voto_i = tf.argmax(votos_models[f"modelo_{i}"], axis=-1)
                voto_suprema = tf.argmax(votos_models["modelo_5"], axis=-1)
                acertou = tf.reduce_all(tf.equal(voto_i, voto_suprema))
                if not acertou:
                    if voto_suprema.shape.rank == 4 and voto_suprema.shape[-1] != 1:
                        voto_suprema = tf.squeeze(voto_suprema, axis=-1)
                        y_ajustado = tf.expand_dims(voto_suprema, axis=-1)
                        assert y_ajustado.shape[-1] == 1, f"y_ajustado tem shape inválido: {y_ajustado.shape}"

                        models[i].fit(x_i,normalizar_y_para_sparse(y_ajustado), epochs=epochs, verbose=0)
                else:
                    assert y_juradas.shape[-1] == 1, f"y_juradas tem shape inválido: {y_juradas.shape}"
                    models[i].fit(x_i, normalizar_y_para_sparse(y_juradas), epochs=epochs, verbose=0)

            else:
                assert y_juradas.shape[-1] == 1, f"y_juradas tem shape inválido: {y_juradas.shape}"
                models[i].fit(x_i, normalizar_y_para_sparse(y_juradas), epochs=epochs, verbose=0)

            votos_models[f"modelo_{i}"] = models[i](x_i, training=False)

        x_juiza = prepare_input_for_model(4, input_tensor_outros)

        stack_suprema = [tf.argmax(votos_models[f"modelo_{i}"], axis=-1) for i in range(5)]
        y_suprema = pixelwise_mode(tf.stack(stack_suprema, axis=0))
        y_suprema = tf.expand_dims(y_suprema, axis=-1)
        x_suprema = prepare_input_for_model(5, input_tensor_outros)
        models[5].fit(x_suprema, normalizar_y_para_sparse(y_suprema), epochs=epochs, verbose=0)
        votos_models["modelo_5"] = models[5](x_suprema, training=False)

        y_juiza = tf.argmax(votos_models["modelo_5"], axis=-1)
        y_juiza = tf.expand_dims(y_juiza, axis=-1)
        log("A Suprema emitiu seu veredito. A Juíza deve reavaliar a causa.")
        models[4].fit(x_juiza, normalizar_y_para_sparse(y_juiza), epochs=epochs, verbose=0)
        votos_models["modelo_4"] = models[4](x_juiza, training=False)

        adv_input = prepare_input_for_model(3, input_tensor_outros)
        y_advogada = tf.argmax(votos_models["modelo_4"], axis=-1)
        y_advogada = tf.expand_dims(y_advogada, axis=-1)
        log("A Juíza respondeu. A Advogada atualiza sua tese com base nesse parecer.")
        models[3].fit(adv_input, normalizar_y_para_sparse(y_advogada), epochs=epochs, verbose=0)
        votos_models["modelo_3"] = models[3](adv_input, training=False)

        gerar_visualizacao_votos(votos_models, input_tensor_outros, idx, block_idx, task_id)

        consenso = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(votos_models["modelo_5"], axis=-1), safe_squeeze(y_suprema, axis=-1)), tf.float32)).numpy()
        log(f"[CONSENSO] Score de consenso com Suprema = {consenso:.3f} (limite = {tol})")

        voto_suprema = tf.argmax(votos_models["modelo_5"], axis=-1)
        for i in range(6):
            voto_i = tf.argmax(votos_models[f"modelo_{i}"], axis=-1)
            acertou = tf.reduce_all(tf.equal(voto_i, voto_suprema)).numpy()
            manager.update_confidence(f"modelo_{i}", bool(acertou))

        manager.log_status(log)

        if consenso >= tol:
            y_eval = tf.argmax(votos_models["modelo_4"], axis=-1)
            y_eval = tf.expand_dims(y_eval, axis=-1)
            loss, acc = models[5].evaluate(x_suprema, normalizar_y_para_sparse(y_eval), verbose=0)
            log(f"[SUPREMA] Avaliação: acc = {acc:.3f}, loss = {loss:.6f}")

            if acc < 1.0 or loss > 0.001:
                log("[SUPREMA] A Suprema rejeitou o veredito da Juíza. Nova deliberação se faz necessária.")
                iter_count += 1
                continue
            else:
                log("[SUPREMA] Confiança plena atingida — consenso final aceito.")
                break

        iter_count += 1

    return {
        "class_logits": votos_models["modelo_5"],
        "consenso": float(consenso)
    }