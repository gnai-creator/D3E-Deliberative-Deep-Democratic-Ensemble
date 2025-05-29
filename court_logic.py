
        # court_logic.py — Sistema de julgamento simbólico com modelos cooperativos e adversariais
import os
import numpy as np
import tensorflow as tf
from confidence_system import avaliar_consenso_ponderado
from runtime_utils import log
from court_utils import extrair_classes_validas, filtrar_classes_respeitando_valores
from metrics_utils import garantir_dict_votos_models
from court_utils import gerar_padrao_simbolico, gerar_visualizacao_votos
from court_utils import inverter_classes_respeitando_valores, pixelwise_mode
from court_utils import treinar_modelo_com_y_sparse, mapear_cores_para_x_test

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

def verificar_votos_models(votos_models):
    for nome, voto in votos_models.items():
        print(f"Modelo: {nome}, Shape: {getattr(voto, 'shape', None)}")

def votos_models_para_lista(votos_models):
    chaves_ordenadas = sorted(votos_models.keys())
    return [votos_models[k] for k in chaves_ordenadas if k in votos_models]

def safe_total_squeeze(t):
    shape = t.shape
    if shape.rank is None:
        return tf.squeeze(t)
    axes = [i for i in range(shape.rank) if shape[i] == 1]
    return tf.squeeze(t, axis=axes)

def arc_court_supreme(models, X_train, y_train, y_val, X_test, task_id=None, block_idx=None,
                      max_cycles=10, tol=0.999, epochs=10, confidence_threshold=0.5,
                      confidence_manager=[], idx=0, pad_value=0, Y_val=None):
    log(f"[SUPREMA] Iniciando deliberação para o bloco {block_idx} — task {task_id}")

    modelos = models.copy()

    def calcular_entropia(y):
        probs = tf.nn.softmax(tf.cast(y, tf.float32), axis=-1)
        log_probs = tf.math.log(tf.clip_by_value(probs, 1e-9, 1.0))
        return -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=-1)).numpy()

    votos_iniciais = {}
    for i in range(5):
        votos_iniciais[f"modelo_{i}"] = modelos[i](X_train, training=False)

    classes_validas = extrair_classes_validas(X_test, pad_value=pad_value)

    for i in range(5):
        votos_iniciais[f"modelo_{i}"] = filtrar_classes_respeitando_valores(
            votos_iniciais[f"modelo_{i}"], classes_validas, pad_value=pad_value
        )

    votos_models = garantir_dict_votos_models(votos_iniciais)
    

    preds_stack = tf.stack(
        [tf.squeeze(tf.argmax(p, axis=-1, output_type=tf.int64), axis=0) for p in votos_models.values()],
        axis=0
    )

    # ⬇️ Usa a verdade objetiva (Y_val) como alvo para Suprema e Promotor
    y_sup = pixelwise_mode(preds_stack)
    y_sup_recolorido = mapear_cores_para_x_test(y_sup, classes_validas)
    y_antitese = inverter_classes_respeitando_valores(y_sup_recolorido, classes_validas, pad_value=pad_value)

    # Agora treina os modelos com rótulos que respeitam as cores do X_test
    treinar_modelo_com_y_sparse(modelos[5], X_test, y_sup_recolorido, epochs=epochs * 3)
    treinar_modelo_com_y_sparse(modelos[6], X_test, y_antitese, epochs=epochs * 3)


    iter_count = 0
    consenso = 0.0

    while iter_count < max_cycles:
        log(f"[DEBUG] iter_count={iter_count}, block_idx={block_idx}, idx={idx}, task_id={task_id}")

        for i in range(7):
            votos_models[f"modelo_{i}"] = modelos[i](X_test, training=False)

        votos_models = garantir_dict_votos_models(votos_models)
        gerar_visualizacao_votos(
            votos_models=votos_models,
            input_tensor_outros=X_train,
            input_tensor_train=X_test,
            iteracao=iter_count,
            idx=idx,
            block_idx=block_idx,
            task_id=task_id
        )
        preds_stack = tf.stack(
            [tf.squeeze(tf.argmax(p, axis=-1, output_type=tf.int64), axis=0) for p in votos_models.values()],
            axis=0
        )

        y_sup = pixelwise_mode(preds_stack)
        y_sup = filtrar_classes_respeitando_valores(y_sup, classes_validas, pad_value=pad_value)
        y_antitese = inverter_classes_respeitando_valores(y_sup, classes_validas, pad_value=pad_value)

        treinar_modelo_com_y_sparse(modelos[5], X_test, y_sup, epochs=epochs)
        treinar_modelo_com_y_sparse(modelos[6], X_test, y_antitese, epochs=epochs)

        for i in range(5):
            y_pred = tf.argmax(modelos[i](X_test, training=False), axis=-1)
            y_target = tf.argmax(y_sup, axis=-1)
            match = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_target), tf.float32)).numpy()
            if match < 0.999:
                treinar_modelo_com_y_sparse(modelos[i], X_test, y_sup, epochs=epochs)

        votos_models = garantir_dict_votos_models(votos_models)
        resultado, consenso = avaliar_consenso_ponderado(
            votos_models,
            pesos,
            required_score=5.0,
            voto_reverso_ok=["modelo_6"]
        )

        if consenso >= tol:
            return {
                "class_logits": votos_models["modelo_5"],
                "consenso": float(consenso),
                "y_pred_simbolico": tf.squeeze(tf.argmax(votos_models["modelo_5"], axis=-1)).numpy()
            }


        iter_count += 1

    return {
        "class_logits": votos_models["modelo_5"],
        "consenso": float(consenso),
        "y_pred_simbolico": tf.squeeze(tf.argmax(votos_models["modelo_5"], axis=-1)).numpy()
    }


