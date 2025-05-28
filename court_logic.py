# court_logic.py — Sistema de julgamento simbólico com modelos cooperativos e adversariais
import os
import numpy as np
import tensorflow as tf
from confidence_system import avaliar_consenso_ponderado
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
    
    modelos = models.copy()
    log(f"MODELOS")

    log(f"[DEBUG] X_test shape: {X_test.shape} | dtype: {X_test.dtype} | min: {tf.reduce_min(X_test).numpy()} | max: {tf.reduce_max(X_test).numpy()}")

    def calcular_entropia(y):
        probs = tf.nn.softmax(tf.cast(y, tf.float32), axis=-1)
        log_probs = tf.math.log(tf.clip_by_value(probs, 1e-9, 1.0))
        return -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=-1)).numpy()

    x_suprema = prepare_input_for_model(5, X_test)
    x_promotor = prepare_input_for_model(6, X_test)

    for i, modelo in enumerate(modelos):
        try:
            _ = modelo.variables
            papel = f"JURADA_{i}" if i in [0, 1, 2] else "ADVOGADO" if i == 3 else "JUIZ" if i == 4 else "SUPREMA" if i == 5 else "PROMOTOR"
            log(f"[DEBUG] {papel} (modelo_{i}) está carregado corretamente.")
        except Exception as e:
            log(f"[ERRO] Modelo_{i} falhou na verificação de integridade: {e}")

    votos_iniciais = {}
    for i in range(5):
        x_i = prepare_input_for_model(i, X_test)
        if i in [0, 3]:
            x_i += tf.random.normal(shape=x_i.shape, mean=0.0, stddev=0.05)
            log(f"[DEBUG] Ruído adicionado ao modelo_{i} ({'JURADA 1' if i == 0 else 'ADVOGADA'})")
        log(f"[DEBUG] modelo_{i} input — shape: {x_i.shape}, min: {tf.reduce_min(x_i).numpy()}, max: {tf.reduce_max(x_i).numpy()}")
        votos_iniciais[f"modelo_{i}"] = modelos[i](x_i, training=False)

    votos_models = votos_iniciais.copy()

    # gerar_visualizacao_votos(votos_iniciais, X_test, idx, block_idx, task_id)

    for i in range(5):
        pred = tf.argmax(votos_iniciais[f"modelo_{i}"], axis=-1)
        valores_unicos = tf.unique(tf.reshape(pred, [-1]))[0].numpy()
        papel = f"JURADA_{i}" if i < 3 else "ADVOGADO" if i == 3 else "JUIZ"
        log(f"[DEBUG] {papel} (modelo_{i}) — valores únicos: {valores_unicos}")

    preds_stack = tf.stack(
        [tf.squeeze(tf.argmax(p, axis=-1, output_type=tf.int64), axis=0) for p in votos_iniciais.values()],
        axis=0
    )
    y_sup = pixelwise_mode(preds_stack)
    if len(np.unique(tf.argmax(y_sup, axis=-1).numpy())) == 1:
        log("[WARN] y_sup colapsado. Inicializando simbolicamente.")
        y_sup = gerar_padrao_simbolico(X_test, pad_value=pad_value)

    log(f"[DEBUG] Entropia média de y_sup: {calcular_entropia(y_sup):.5f}")

    treinar_modelo_com_y_sparse(modelos[5], x_suprema, y_sup, epochs=epochs * 3)

    classes_validas = extrair_classes_validas(X_test, pad_value=pad_value)
    y_antitese = inverter_classes_respeitando_valores(y_sup, classes_validas, pad_value=pad_value)
    log("[PROMOTOR] Treinando promotor com antítese da Suprema.")
    treinar_modelo_com_y_sparse(modelos[6], x_promotor, y_antitese, epochs=epochs)

    iter_count = 0
    consenso = 0.0

    while iter_count < max_cycles:
        skip_reeducacao = iter_count == 0
        if skip_reeducacao:
            log("[JURADOS] Modo teimoso ativado — nenhum modelo será retreinado no primeiro ciclo.")
        log(f"[CICLO] Iteração {iter_count}")

        for i in range(5, 7):
            x_i = prepare_input_for_model(i, X_test)
            votos_models[f"modelo_{i}"] = modelos[i](x_i, training=False)

        # if iter_count == 0:
        #     gerar_visualizacao_votos(votos_iniciais, X_test, idx, block_idx, task_id)
        # else:
        gerar_visualizacao_votos(votos_models, X_test, idx, block_idx, task_id)
        if iter_count > 0:
            for i in range(5):
                x_i = prepare_input_for_model(i, X_test)
                votos_models[f"modelo_{i}"] = modelos[i](x_i, training=False)

        votos_models = garantir_dict_votos_models(votos_models)

        juradas_preds = [votos_models[f"modelo_{i}"] for i in range(3)]
        juradas_classes = tf.stack(
            [tf.squeeze(tf.argmax(p, axis=-1, output_type=tf.int64), axis=0) for p in juradas_preds],
            axis=0
        )
        log(f"[DEBUG] Classes juradas únicas: {np.unique(juradas_classes.numpy())}")
        y_sup = pixelwise_mode(juradas_classes)
        log(f"[DEBUG] y_sup unique values: {np.unique(tf.argmax(y_sup, axis=-1).numpy())}")

        log(f"[DEBUG] Entropia média de y_sup (iter {iter_count}): {calcular_entropia(y_sup):.5f}")

        treinar_modelo_com_y_sparse(modelos[5], x_suprema, y_sup, epochs=epochs)

        for i in range(5):
            x_i = prepare_input_for_model(i, X_test)
            y_pred = tf.argmax(modelos[i](x_i, training=False), axis=-1, output_type=tf.int64)
            y_pred = tf.expand_dims(y_pred, axis=-1)
            y_pred_corrigido = filtrar_classes_respeitando_valores(y_pred, classes_validas, pad_value=pad_value)
            y_sup_argmax = tf.expand_dims(tf.argmax(y_sup, axis=-1), axis=-1)
            y_pred_argmax = tf.argmax(y_pred_corrigido, axis=-1)

            papel = f"JURADA_{i}" if i in [0, 1, 2] else "ADVOGADO" if i == 3 else "JUIZ"

            match = tf.reduce_mean(
                tf.cast(tf.equal(y_pred_argmax, tf.squeeze(y_sup_argmax, axis=-1)), tf.float32)
            ).numpy()
            log(f"[DEBUG] Match modelo_{i} vs Suprema: {match:.3f}")
            if not skip_reeducacao and match < 0.99:
                log(f"[REEDUCAÇÃO] Retreinando modelo_{i} ({papel}) — match: {match:.3f}, max class: {tf.reduce_max(y_pred_argmax).numpy()}")
                log(f"[DEBUG] y_sup valores únicos antes do treino modelo_{i}: {np.unique(tf.argmax(y_sup, axis=-1).numpy())}")
                treinar_modelo_com_y_sparse(modelos[i], x_i, y_sup, epochs=epochs)
            else:
                log(f"[ALINHADO] {papel} (modelo_{i}) está em acordo com Suprema ({match:.3f})")

        y_antitese = inverter_classes_respeitando_valores(y_sup, classes_validas, pad_value=pad_value)
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
