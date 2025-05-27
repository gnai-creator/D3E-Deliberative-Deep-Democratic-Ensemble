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


def extrair_classes_validas(y_real, pad_value=0):
    y_real = tf.convert_to_tensor(y_real)
    log(f"[DEBUG] extrair_classes_validas — y_real.shape={y_real.shape}")
    if y_real.shape.rank == 4:
        y_real = tf.squeeze(y_real, axis=0)
    if y_real.shape.rank == 3 and y_real.shape[-1] == 1:
        y_real = tf.squeeze(y_real, axis=-1)
    valores = tf.unique(tf.reshape(y_real, [-1]))[0]
    valores = tf.cast(valores, tf.int32)
    return tf.boolean_mask(valores, valores != pad_value)

def inverter_classes_respeitando_valores(y, classes_validas, pad_value=0):
    y = tf.convert_to_tensor(y)
    log(f"[DEBUG] inverter_classes — y.shape={y.shape}, classes_validas={classes_validas.numpy()}")

    # Padroniza para (H, W)
    if y.shape.rank == 4:
        y = tf.squeeze(y, axis=0)
    if y.shape.rank == 3 and y.shape[-1] == 1:
        y = tf.squeeze(y, axis=-1)
    elif y.shape.rank == 3:
        y = y[0]  # assume (1, H, W)

    y_flat = tf.reshape(y, [-1])
    classes_validas = tf.reshape(classes_validas, [-1])
    classes_validas = tf.boolean_mask(classes_validas, classes_validas != pad_value)
    if tf.size(classes_validas) == 0:
        antitese = y_flat
    else:
        classes_validas = tf.reshape(classes_validas, [-1, 1])
        diffs = tf.abs(tf.cast(classes_validas, tf.int32) - tf.cast(y_flat, tf.int32))
        idx_max = tf.argmax(diffs, axis=0)
        antitese = tf.gather(tf.reshape(classes_validas, [-1]), idx_max)

    antitese = tf.reshape(antitese, tf.shape(y))  # (H, W)
    antitese = tf.expand_dims(antitese, axis=-1)  # (H, W, 1)
    antitese = tf.expand_dims(antitese, axis=0)   # (1, H, W, 1)
    return antitese

def filtrar_classes_respeitando_valores(y, classes_validas, pad_value=0):
    y = tf.convert_to_tensor(y)
    log(f"[DEBUG] filtrando classes — y.shape={y.shape}, classes_validas={classes_validas.numpy()}")

    # Ajusta para formato comum (H, W)
    if y.shape.rank == 4:
        y = tf.squeeze(y, axis=0)  # (H, W, C)
    if y.shape.rank == 3 and y.shape[-1] == 1:
        channel = y[..., 0]  # (H, W)
    elif y.shape.rank == 3:
        channel = y[0]  # assume (1, H, W)
    elif y.shape.rank == 2:
        channel = y
    else:
        raise ValueError(f"[filtrar_classes_respeitando_valores] Shape inesperado: {y.shape}")

    # Filtra classes
    mask = tf.reduce_any(tf.equal(channel[..., tf.newaxis], tf.cast(classes_validas, channel.dtype)), axis=-1)
    filtrado = tf.where(mask, channel, tf.constant(pad_value, dtype=channel.dtype))  # (H, W)

    # Reconstrói shape (1, H, W, 1)
    filtrado = tf.expand_dims(filtrado, axis=-1)  # (H, W, 1)
    filtrado = tf.expand_dims(filtrado, axis=0)   # (1, H, W, 1)
    return filtrado


def arc_court_supreme(models, input_tensor_outros, task_id=None, block_idx=None,
                      max_cycles=10, tol=0.98, epochs=10, confidence_threshold=0.5,
                      confidence_manager=[], idx=0, pad_value=0, Y_val=None):
    log(f"[SUPREMA] Iniciando deliberação para o bloco {block_idx} — task {task_id}")
    votos_models = {}
    modelos = models.copy()

    for i in range(7):
        x_i = prepare_input_for_model(i, input_tensor_outros)
        votos_models[f"modelo_{i}"] = modelos[i](x_i, training=False)

    votos_models = garantir_dict_votos_models(votos_models)

    if Y_val is None:
        raise ValueError("[SUPREMA] Y_val (resposta esperada) é obrigatório para extrair as cores válidas da tarefa.")
    classes_validas = extrair_classes_validas(Y_val, pad_value=pad_value)

    # Atualiza voto da Suprema antes do treino do promotor
    juradas_preds = [votos_models[f"modelo_{i}"] for i in range(3)]
    juradas_classes = tf.stack([tf.argmax(p, axis=-1, output_type=tf.int64) for p in juradas_preds], axis=0)
    y_sup = pixelwise_mode(juradas_classes)
    y_sup_corrigido = filtrar_classes_respeitando_valores(y_sup, classes_validas, pad_value=pad_value)
    y_suprema = tf.expand_dims(y_sup_corrigido, axis=-1)
    x_suprema = prepare_input_for_model(5, input_tensor_outros)
    modelos[5].fit(x=x_suprema, y=y_suprema, epochs=epochs, verbose=0)

    # Promotor treinado com antítese coerente
    x_promotor = prepare_input_for_model(6, input_tensor_outros)
    y_antitese = inverter_classes_respeitando_valores(y_sup_corrigido, classes_validas, pad_value=pad_value)
    y_antitese = tf.expand_dims(y_antitese, axis=-1)
    log("[PROMOTOR] Treinando promotor com antítese da Suprema.")
    modelos[6].fit(x=x_promotor, y=y_antitese, epochs=epochs, verbose=0)

    iter_count = 0
    consenso = 0.0

    while iter_count < max_cycles:
        log(f"[CICLO] Iteração {iter_count}")

        for i in range(6):
            x_i = prepare_input_for_model(i, input_tensor_outros)
            votos_models[f"modelo_{i}"] = modelos[i](x_i, training=False)

        votos_models = garantir_dict_votos_models(votos_models)
        gerar_visualizacao_votos(votos_models, input_tensor_outros, idx, block_idx, task_id)

        # Suprema
        juradas_preds = [votos_models[f"modelo_{i}"] for i in range(3)]
        juradas_classes = tf.stack([tf.argmax(p, axis=-1, output_type=tf.int64) for p in juradas_preds], axis=0)
        y_sup = pixelwise_mode(juradas_classes)
        y_sup_corrigido = filtrar_classes_respeitando_valores(y_sup, classes_validas, pad_value=pad_value)
        y_suprema = tf.expand_dims(y_sup_corrigido, axis=-1)

        x_suprema = prepare_input_for_model(5, input_tensor_outros)
        modelos[5].fit(x=x_suprema, y=y_suprema, epochs=epochs, verbose=0)

        # Reeducar modelos 0 a 4
        for i in range(5):
            x_i = prepare_input_for_model(i, input_tensor_outros)
            y_pred = tf.argmax(modelos[i](x_i, training=False), axis=-1, output_type=tf.int64)
            y_pred = tf.expand_dims(y_pred, axis=-1)
            y_pred_corrigido = filtrar_classes_respeitando_valores(y_pred, classes_validas, pad_value=pad_value)
            match = tf.reduce_mean(tf.cast(tf.equal(y_pred_corrigido, tf.cast(y_suprema, tf.int64)), tf.float32)).numpy()
            if match < 0.95:
                log(f"[REEDUCAÇÃO] Modelo_{i} em desacordo com Suprema ({match:.3f}) - retreinando...")
                modelos[i].fit(x=x_i, y=y_suprema, epochs=epochs, verbose=0)
            else:
                log(f"[ALINHADO] Modelo_{i} está em acordo com Suprema ({match:.3f})")

        # Promotor
        x_promotor = prepare_input_for_model(6, input_tensor_outros)
        y_sup_squeezed = tf.squeeze(y_suprema)
        if y_sup_squeezed.shape.rank == 2:
            y_sup_squeezed = tf.expand_dims(y_sup_squeezed, axis=0)

        y_antitese = inverter_classes_respeitando_valores(y_sup_squeezed, classes_validas, pad_value=pad_value)
        y_antitese = tf.expand_dims(y_antitese, axis=-1)

        log("[PROMOTOR] Propondo antítese ao parecer da Suprema.")
        modelos[6].fit(x=x_promotor, y=y_antitese, epochs=epochs, verbose=0)

        # Checagem de consenso simbólico
        votos_models = garantir_dict_votos_models(votos_models)
        consenso = avaliar_consenso_com_confianca(
            votos_models,
            confidence_manager,
            required_votes=3,
            confidence_threshold=confidence_threshold,
            voto_reverso_ok=["modelo_6"]
        )

        if consenso >= tol:
            log("[SUPREMA] Consenso atingido. Encerrando deliberação.")
            return {
                "class_logits": votos_models["modelo_5"],
                "consenso": float(consenso)
            }

        iter_count += 1

    log("[SUPREMA] Deliberação encerrada sem consenso total.")
    return {
        "class_logits": votos_models["modelo_5"],
        "consenso": float(consenso)
    }
