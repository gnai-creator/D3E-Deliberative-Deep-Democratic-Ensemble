import os
import numpy as np
import tensorflow as tf
from confidence_system import avaliar_consenso_com_confianca
from metrics_utils import salvar_voto_visual, preparar_voto_para_visualizacao
from runtime_utils import log
from metrics_utils import safe_squeeze, ensure_numpy
from models_loader import load_model

# Converte logits para rótulos se necessário
def normalizar_y_para_sparse(y):
    log(f"[DEBUG] Y SHAPE: {y.shape}")
    if y.shape.rank == 4 and y.shape[-1] != 1:
        y = tf.argmax(y, axis=-1)
        y = tf.expand_dims(y, axis=-1)
    return y

def garantir_dict_votos_models(votos_models):
    if isinstance(votos_models, dict):
        return votos_models
    elif isinstance(votos_models, list):
        return {f"modelo_{i}": v for i, v in enumerate(votos_models)}
    else:
        log(f"[SECURITY] votos_models tinha tipo inesperado: {type(votos_models)}. Substituindo por dict vazio.")
        return {}

def pixelwise_mode(stack):
    stack = tf.transpose(stack, [1, 2, 3, 0])
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
    votos_models = garantir_dict_votos_models(votos_models)
    votos_visuais = []
    try:
        for v in votos_models.values():
            log(f"[DEBUG] preparando voto: type={type(v)}, shape={getattr(v, 'shape', 'indefinido')}")
            resultado = preparar_voto_para_visualizacao(v)
            if resultado is not None:
                votos_visuais.append(resultado)
    except Exception as e:
        log(f"[VISUAL] Erro ao processar votos_models: {e}")
    try:
        input_tensor_outros = ensure_numpy(input_tensor_outros)
        if input_tensor_outros.ndim == 5:
            input_visual = input_tensor_outros[0, :, :, 0, 0]
        elif input_tensor_outros.ndim == 4:
            input_visual = input_tensor_outros[0, :, :, 0]
        elif input_tensor_outros.ndim == 3:
            input_visual = input_tensor_outros[0, :, :]
        else:
            raise ValueError("Shape inesperado")
        if input_visual.ndim != 2:
            raise ValueError("input_visual nao e 2D")
    except Exception as e:
        log(f"[VISUAL] input_visual com shape inesperado ({getattr(input_tensor_outros, 'shape', 'N/A')}): {e}")
        input_visual = tf.zeros((30, 30), dtype=tf.int32)
    salvar_voto_visual(votos_visuais, idx, block_idx, input_visual, task_id=task_id)

def treinar_promotor_inicial(models, input_tensor_outros, votos_models, epochs):
    votos_models = garantir_dict_votos_models(votos_models)
    juradas_preds = [votos_models[f"modelo_{i}"] for i in range(3)]
    juradas_classes = tf.stack([tf.argmax(p, axis=-1) for p in juradas_preds], axis=0)
    y_moda = pixelwise_mode(juradas_classes)
    y_antitese = 9 - y_moda
    y_antitese = tf.clip_by_value(y_antitese, 0, 9)
    y_antitese = tf.expand_dims(y_antitese, axis=-1)

    x_promotor = prepare_input_for_model(6, input_tensor_outros)
    log(f"[DEBUG] x_promotor shape: {x_promotor.shape}")
    log(f"[DEBUG] y_antitese shape: {y_antitese.shape}")
    log("[PROMOTOR] Treinando promotor com antítese da moda das juradas.")
    models[6].fit(x=x_promotor, y=y_antitese, epochs=epochs, verbose=0)

def instanciar_promotor_e_supremo(models):
    model = load_model(6, 0.0005)
    models.append(model)
    return models

def arc_court_supreme(models, input_tensor_outros, task_id=None, block_idx=None,
                      max_cycles=10, tol=0.98, epochs=10, confidence_threshold=0.5,
                      confidence_manager=[], idx=0):
    log(f"[SUPREMA] Iniciando deliberação para o bloco {block_idx} — task {task_id}")
    votos_models = {}
    modelos = instanciar_promotor_e_supremo(models)

    for i in range(7):
        x_i = prepare_input_for_model(i, input_tensor_outros)
        votos_models[f"modelo_{i}"] = modelos[i](x_i, training=False)

    votos_models = garantir_dict_votos_models(votos_models)
    treinar_promotor_inicial(modelos, input_tensor_outros, votos_models, epochs)

    iter_count = 0
    consenso = 0.0

    while iter_count < max_cycles:
        log(f"[CICLO] Iteração {iter_count}")
        for i in range(7):
            x_i = prepare_input_for_model(i, input_tensor_outros)
            votos_models[f"modelo_{i}"] = modelos[i](x_i, training=False)

        votos_models = garantir_dict_votos_models(votos_models)

        if "modelo_3" not in votos_models:
            log("[CORTE] modelo_3 ainda não está disponível. Pulando rodada.")
            return {"consenso": 0.0, "votos_models": votos_models}

        y_juradas = tf.argmax(votos_models["modelo_3"], axis=-1)
        y_juradas = tf.expand_dims(y_juradas, axis=-1)
        log(f"[DEBUG] y_juradas shape: {y_juradas.shape}")

        if iter_count == 0:
            gerar_visualizacao_votos(votos_models, input_tensor_outros, idx, block_idx, task_id)
            iter_count += 1
            continue

        votos_models = garantir_dict_votos_models(votos_models)
        juradas_preds = [votos_models[f"modelo_{i}"] for i in range(3)]
        juradas_classes = tf.stack([tf.argmax(p, axis=-1) for p in juradas_preds], axis=0)
        y_suprema = pixelwise_mode(juradas_classes)
        y_suprema = tf.expand_dims(y_suprema, axis=-1)

        x_suprema = prepare_input_for_model(5, input_tensor_outros)
        modelos[5].fit(x=x_suprema, y=y_suprema, epochs=epochs, verbose=0)

        # for i in range(5):
        #     x_i = prepare_input_for_model(i, input_tensor_outros)
        #     y_pred = tf.argmax(modelos[i](x_i, training=False), axis=-1)
        #     y_pred = tf.expand_dims(y_pred, axis=-1)
        #     match = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_suprema), tf.float32)).numpy()
        #     if match < 0.95:
        #         log(f"[REEDUCAÇÃO] Modelo_{i} está em desacordo com Suprema ({match:.3f}) — retreinando...")
        #         modelos[i].fit(x=x_i, y=y_suprema, epochs=epochs, verbose=0)
        #     else:
        #         log(f"[ALINHADO] Modelo_{i} já está de acordo com Suprema ({match:.3f})")

        x_promotor = prepare_input_for_model(6, input_tensor_outros)
        y_sup = tf.squeeze(y_suprema)
        if y_sup.shape.rank == 2:
            y_sup = tf.expand_dims(y_sup, axis=0)
        y_antitese = 9 - y_sup
        y_antitese = tf.clip_by_value(y_antitese, 0, 9)
        y_antitese = tf.expand_dims(y_antitese, axis=-1)
        log("[PROMOTOR] Propondo antítese ao parecer da Suprema.")
        modelos[6].fit(x=x_promotor, y=y_antitese, epochs=epochs, verbose=0)

        promotor_pred = tf.argmax(modelos[6](x_promotor, training=False), axis=-1)
        juradas_consenso = pixelwise_mode(tf.stack([tf.argmax(votos_models[f"modelo_{i}"], axis=-1) for i in range(3)], axis=0))
        promotor_agreement = tf.reduce_mean(tf.cast(tf.equal(promotor_pred, juradas_consenso), tf.float32)).numpy()

        if promotor_agreement > 0.9:
            log("[PROMOTOR] Promotor está alinhado com juradas — Suprema deve reconsiderar.")
            modelos[5].fit(x=x_suprema, y=juradas_consenso[..., tf.newaxis], epochs=epochs, verbose=0)

        gerar_visualizacao_votos(votos_models, input_tensor_outros, idx, block_idx, task_id)

        votos_models = garantir_dict_votos_models(votos_models)
        consenso = avaliar_consenso_com_confianca(
            votos_models,
            confidence_manager,
            required_votes=4,
            confidence_threshold=0.5,
            voto_reverso_ok=["modelo_6"]
        )
 

        if consenso >= tol:
            # Conversão segura dos logits da Juíza
            y_juiza_logits = votos_models["modelo_4"]
            if y_juiza_logits.shape[-1] > 1:
                y_juiza = tf.argmax(y_juiza_logits, axis=-1)
            else:
                y_juiza = tf.squeeze(y_juiza_logits)

            y_suprema_pred = tf.argmax(modelos[5](x_suprema), axis=-1)
            y_promotor_pred = tf.argmax(modelos[6](x_promotor), axis=-1)
            y_promotor_corrigido = tf.clip_by_value(9 - y_promotor_pred, 0, 9)

            # Cast explícito para garantir que todos sejam int64
            y_juiza = tf.cast(y_juiza, tf.int64)
            y_suprema_pred = tf.cast(y_suprema_pred, tf.int64)
            y_promotor_pred = tf.cast(y_promotor_pred, tf.int64)
            y_promotor_corrigido = tf.cast(y_promotor_corrigido, tf.int64)

            acordo_juiza_suprema = tf.reduce_all(tf.equal(y_suprema_pred, y_juiza)).numpy()
            acordo_promotor_juiza = tf.reduce_all(tf.equal(y_promotor_corrigido, y_juiza)).numpy()
            acordo_promotor_suprema = tf.reduce_all(tf.equal(y_promotor_pred, y_suprema_pred)).numpy()

            log(f"[CHECK] Suprema == Juíza? {acordo_juiza_suprema}")
            log(f"[CHECK] Promotor corrigido == Juíza? {acordo_promotor_juiza}")
            log(f"[CHECK] Promotor == Suprema? {acordo_promotor_suprema}")

            if acordo_juiza_suprema and acordo_promotor_juiza and acordo_promotor_suprema:
                log("[SUPREMA] Suprema, Juíza e Promotor (corrigido) estão em acordo literal. Prosseguindo para próximo bloco.")
                break
            else:
                log("[SUPREMA] Divergência detectada entre Juíza, Suprema ou Promotor. Nova deliberação se faz necessária.")
                iter_count += 1
                continue




        iter_count += 1

    return {
        "class_logits": votos_models["modelo_5"],
        "consenso": float(consenso)
    }