# court_logic.py — Sistema de julgamento simbólico com modelos cooperativos e adversariais
import os
import numpy as np
import tensorflow as tf
from confidence_system import avaliar_consenso_com_confianca, avaliar_consenso_ponderado
from runtime_utils import log
from court_utils import extrair_classes_validas, filtrar_classes_respeitando_valores
from court_utils import prepare_input_for_model, garantir_dict_votos_models
from court_utils import gerar_padrao_simbolico, gerar_visualizacao_votos
from court_utils import inverter_classes_respeitando_valores, pixelwise_mode
from court_utils import treinar_modelo_com_y_sparse

# Ativa modo eager para debug detalhado de train_function
tf.config.run_functions_eagerly(True)

pesos = {
    "modelo_0": 1.0,
    "modelo_1": 1.0,
    "modelo_2": 1.0,
    "modelo_3": 1.5,
    "modelo_4": 2.0,
    "modelo_5": 3.0,
    "modelo_6": -2.0
}

def safe_total_squeeze(t):
    shape = t.shape
    if shape.rank is None:
        return tf.squeeze(t)
    axes = [i for i in range(shape.rank) if shape[i] == 1]
    return tf.squeeze(t, axis=axes)

def arc_court_supreme(models, X_test, task_id=None, block_idx=None,
                      max_cycles=10, tol=0.98, epochs=10, confidence_threshold=0.5,
                      confidence_manager=[], idx=0, pad_value=0, Y_val=None):
    log(f"[SUPREMA] Iniciando deliberação para o bloco {block_idx} — task {task_id}")
    votos_models = {}
    modelos = models.copy()
    log(f"MODELOS")

    x_juiza = prepare_input_for_model(4, X_test)
    x_suprema = prepare_input_for_model(5, X_test)
    x_promotor = prepare_input_for_model(6, X_test)

    for i, x_input in zip([4, 5, 6], [x_juiza, x_suprema, x_promotor]):
        try:
            output_dim = 40 if x_input.shape[-1] == 40 else 4
            dummy_input = tf.random.uniform((1, 30, 30, 10, output_dim), dtype=tf.float32)
            log(f"[DEBUG] shape dummy_input modelo_{i}: {dummy_input.shape}")
            output_sample = modelos[i](dummy_input, training=False)
            log(f"[DEBUG] modelo_{i} ativado com output shape: {output_sample.shape}")
        except Exception as e:
            log(f"[ERRO] Falha ao ativar modelo_{i} com dummy_input: {e}")

    for i in range(7):
        x_i = prepare_input_for_model(i, X_test)
        votos_models[f"modelo_{i}"] = modelos[i](x_i, training=False)

    log(f"[DEBUG] VOTOS MODELS SHAPE: {len(votos_models)}")
    votos_models = garantir_dict_votos_models(votos_models)
    classes_validas = extrair_classes_validas(X_test, pad_value=pad_value)

    y_sup = gerar_padrao_simbolico(X_test)
    y_sup = safe_total_squeeze(y_sup)
    y_sup = tf.cast(y_sup, tf.int32)
    y_sup = tf.expand_dims(y_sup, axis=0)
    y_sup = tf.expand_dims(y_sup, axis=-1)

    num_classes_juizas = modelos[4].output_shape[-1]
    num_classes_suprema = modelos[5].output_shape[-1]
    num_classes_promotor = modelos[6].output_shape[-1]

    y_suprema = tf.one_hot(y_sup, depth=num_classes_suprema, axis=-1)
    y_suprema = tf.cast(y_suprema, tf.float32)

    log(f"[DEBUG] Suprema valores únicos: {tf.unique(tf.reshape(tf.argmax(y_suprema, axis=-1), [-1]))[0].numpy()}")
    log(f"[DEBUG] y_suprema (simbolico) shape: {y_suprema.shape}")

    treinar_modelo_com_y_sparse(modelos[5], x_suprema, y_suprema, epochs=epochs * 3)

    y_antitese = inverter_classes_respeitando_valores(tf.squeeze(y_sup, axis=0), classes_validas, pad_value=pad_value)
    y_antitese = tf.expand_dims(y_antitese, axis=0)
    y_antitese = tf.expand_dims(y_antitese, axis=-1)
    y_antitese = tf.one_hot(y_antitese, depth=num_classes_promotor, axis=-1)
    y_antitese = tf.cast(y_antitese, tf.float32)

    log("[PROMOTOR] Treinando promotor com antítese da Suprema.")
    treinar_modelo_com_y_sparse(modelos[6], x_promotor, y_antitese, epochs=epochs)

    iter_count = 0
    consenso = 0.0

    while iter_count < max_cycles:
        log(f"[CICLO] Iteração {iter_count}")

        for i in range(6):
            x_i = prepare_input_for_model(i, X_test)
            votos_models[f"modelo_{i}"] = modelos[i](x_i, training=False)

        votos_models = garantir_dict_votos_models(votos_models)
        gerar_visualizacao_votos(votos_models, X_test, idx, block_idx, task_id)

        juradas_preds = [votos_models[f"modelo_{i}"] for i in range(3)]
        juradas_classes = tf.stack([tf.argmax(p, axis=-1, output_type=tf.int64) for p in juradas_preds], axis=0)
        y_sup = pixelwise_mode(juradas_classes)
        y_sup = safe_total_squeeze(y_sup)
        y_sup = tf.cast(y_sup, tf.int32)
        y_sup = tf.expand_dims(y_sup, axis=0)
        y_sup = tf.expand_dims(y_sup, axis=-1)
        y_suprema = tf.one_hot(y_sup, depth=num_classes_juizas, axis=-1)
        y_suprema = tf.cast(y_suprema, tf.float32)

        treinar_modelo_com_y_sparse(modelos[5], x_suprema, y_suprema, epochs=epochs)

        for i in range(5):
            x_i = prepare_input_for_model(i, X_test)
            y_pred = tf.argmax(modelos[i](x_i, training=False), axis=-1, output_type=tf.int64)
            y_pred = tf.expand_dims(y_pred, axis=-1)
            y_pred_corrigido = filtrar_classes_respeitando_valores(y_pred, classes_validas, pad_value=pad_value)
            match = tf.reduce_mean(tf.cast(tf.equal(y_pred_corrigido, tf.argmax(y_suprema, axis=-1)), tf.float32)).numpy()
            if match < 0.95:
                log(f"[REEDUCAÇÃO] Modelo_{i} em desacordo com Suprema ({match:.3f}) - retreinando...")
                treinar_modelo_com_y_sparse(modelos[i], x_i, y_suprema, epochs=epochs)
            else:
                log(f"[ALINHADO] Modelo_{i} está em acordo com Suprema ({match:.3f})")

        y_antitese = inverter_classes_respeitando_valores(tf.squeeze(y_sup, axis=0), classes_validas, pad_value=pad_value)
        y_antitese = tf.expand_dims(y_antitese, axis=0)
        y_antitese = tf.expand_dims(y_antitese, axis=-1)
        y_antitese = tf.one_hot(y_antitese, depth=num_classes_promotor, axis=-1)
        y_antitese = tf.cast(y_antitese, tf.float32)

        log("[PROMOTOR] Propondo antítese ao parecer da Suprema.")
        treinar_modelo_com_y_sparse(modelos[6], x_promotor, y_antitese, epochs=epochs)

        votos_models = garantir_dict_votos_models(votos_models)
        consenso = avaliar_consenso_ponderado(
            votos_models,
            pesos,
            required_score=5.0,
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