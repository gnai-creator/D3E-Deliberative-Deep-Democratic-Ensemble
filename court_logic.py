
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
from court_utils import extrair_todas_classes_validas
from court_utils import extrair_canal_cor, expandir_para_3_canais
# Ativa modo eager para debug detalhado de train_function
tf.config.run_functions_eagerly(True)

pesos = {
    "modelo_0": 1.0,
    "modelo_1": 1.0,
    "modelo_2": 1.0,
    "modelo_3": 1.5,
    "modelo_4": 2.0,
    "modelo_5": 3.0,
    "modelo_6": -3.5
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
                      max_cycles=20, tol=9.0, epochs=10, confidence_threshold=0.5,
                      confidence_manager=[], idx=0, pad_value=-1, Y_val=None):
    log(f"[SUPREMA] Iniciando deliberação para o bloco {block_idx} — task {task_id}")

    modelos = models.copy()


    votos_iniciais = {}
    
    votos_iniciais[f"modelo_{0}"] = modelos[0](X_train, training=False)
    votos_iniciais[f"modelo_{1}"] = modelos[1](X_train, training=False)
    votos_iniciais[f"modelo_{2}"] = modelos[2](X_train, training=False)
    votos_iniciais[f"modelo_{3}"] = modelos[3](X_train, training=False)
    votos_iniciais[f"modelo_{4}"] = modelos[4](X_train, training=False)
    classes_validas = extrair_todas_classes_validas(X_test, X_train, pad_value=pad_value)
    classes_objetivo = extrair_classes_validas(X_test, pad_value=pad_value)

    # for i in range(5):
    #     votos_iniciais[f"modelo_{i}"] = filtrar_classes_respeitando_valores(
    #         votos_iniciais[f"modelo_{i}"], classes_validas, pad_value=pad_value
    #     )

    # Mantém os votos fixos dos jurados e advogada

    votos_models = {}
    
    for i in range(1, 5):
        votos_models[f"modelo_{i}"] = votos_iniciais[f"modelo_{i}"]

    for nome, voto in votos_models.items():
        if voto is None:
            log(f"[AVISO] Voto de {nome} está None.")
        elif not isinstance(voto, tf.Tensor):
            log(f"[AVISO] Voto de {nome} não é tensor: {type(voto)}")
        else:
            log(f"[DEBUG] {nome} voto shape: {voto.shape}")

    # votos_models = garantir_dict_votos_models(votos_iniciais)
    preds_stack = tf.stack(
        [
            tf.expand_dims(
                tf.cast(
                    tf.squeeze(tf.argmax(extrair_canal_cor(p), axis=-1, output_type=tf.int64), axis=0),
                    tf.int32
                ),
                axis=-1
            )
            for p in votos_models.values() if isinstance(p, tf.Tensor)
        ],
        axis=0
    )

    # ⬇️ Usa a verdade objetiva (Y_val) como alvo para Suprema e Promotor
    y_sup = pixelwise_mode(preds_stack)
    y_sup_recolorido = mapear_cores_para_x_test(y_sup, classes_validas)
    y_antitese = inverter_classes_respeitando_valores(y_sup_recolorido, classes_validas, pad_value=pad_value)

    y_sup_redi = expandir_para_3_canais(y_sup_recolorido)
    y_antitese_redi = expandir_para_3_canais(y_antitese)
    # Agora treina os modelos com rótulos que respeitam as cores do X_test
    treinar_modelo_com_y_sparse(modelos[5], X_test, y_sup_redi, epochs=epochs * 3)
    treinar_modelo_com_y_sparse(modelos[6], X_test, y_antitese_redi, epochs=epochs * 3)

    log(f"[DEBUG] Classes únicas após filtragem modelo_{i}: {np.unique(votos_iniciais[f'modelo_{i}'].numpy())}")

    iter_count = 0
    consenso = 0.0

    while iter_count < max_cycles:
        log(f"[DEBUG] iter_count={iter_count}, block_idx={block_idx}, idx={idx}, task_id={task_id}")
        
        if iter_count >= max_cycles/4:
            votos_models[f"modelo_{0}"] = modelos[0](X_test, training=False)
        # Garante que os votos dos modelos 1–4 sejam mantidos ao longo do loop
        for i in range(1, 5):
            votos_models[f"modelo_{i}"] = votos_iniciais[f"modelo_{i}"]

        
        votos_models[f"modelo_{5}"] = modelos[5](X_test, training=False)
        votos_models[f"modelo_{6}"] = modelos[6](X_test, training=False)

        votos_models = garantir_dict_votos_models(votos_models)


        iter_count += 1

        preds_stack = tf.stack(
            [
                tf.expand_dims(
                    tf.cast(
                        tf.squeeze(tf.argmax(extrair_canal_cor(p), axis=-1, output_type=tf.int64), axis=0),
                        tf.int32
                    ),
                    axis=-1
                )
                for p in votos_models.values() if isinstance(p, tf.Tensor)
            ],
            axis=0
        )

        y_sup = pixelwise_mode(preds_stack)
        y_sup_recolorido = mapear_cores_para_x_test(y_sup, classes_validas)
        y_antitese = inverter_classes_respeitando_valores(y_sup_recolorido, classes_validas, pad_value=pad_value)

        y_sup_redi = expandir_para_3_canais(y_sup_recolorido)
        y_antitese_redi = expandir_para_3_canais(y_antitese)
        # Agora treina os modelos com rótulos que respeitam as cores do X_test
        treinar_modelo_com_y_sparse(modelos[5], X_test, y_sup_redi, epochs=epochs * 3)
        treinar_modelo_com_y_sparse(modelos[6], X_test, y_antitese_redi, epochs=epochs * 3)

       
        if iter_count >= max_cycles / 4:
            y_pred = tf.argmax(modelos[0](X_test, training=False), axis=-1)
            y_target = tf.argmax(y_sup, axis=-1)
            match = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_target), tf.float32)).numpy()
            log(f"[MATCH] MATCH {match}")
            if match < 0.97:
                y_sup_recolorido = mapear_cores_para_x_test(y_sup, classes_objetivo)
                treinar_modelo_com_y_sparse(modelos[0], X_test, y_sup_recolorido, epochs=epochs)
        

        votos_models = garantir_dict_votos_models(votos_models)
        resultado, consenso = avaliar_consenso_ponderado(
            votos_models,
            pesos,
            required_score=5.0,
            voto_reverso_ok=["modelo_6"]
        )

        gerar_visualizacao_votos(
            votos_models=votos_models,
            input_tensor_outros=X_train,
            input_tensor_train=X_test,
            iteracao=iter_count,
            idx=idx,
            block_idx=block_idx,
            task_id=task_id,
            classes_validas=classes_validas,
            classes_objetivo=classes_objetivo,
            consenso=consenso
        )
        
        log(f"[CONSENSO] : CONSENSO {consenso}")
        if consenso >= tol:
            return {
                "class_logits": votos_models["modelo_5"],
                "consenso": float(consenso),
                "y_pred_simbolico": tf.squeeze(tf.argmax(votos_models["modelo_5"], axis=-1)).numpy()
            }


    
    for model_idx in range(len(models)):
        models[model_idx].save_weights(f"weights_model_{model_idx}_block_{block_idx}.h5")

    return {
        "class_logits": votos_models["modelo_5"],
        "consenso": float(consenso),
        "y_pred_simbolico": tf.squeeze(tf.argmax(votos_models["modelo_5"], axis=-1)).numpy()
    }
