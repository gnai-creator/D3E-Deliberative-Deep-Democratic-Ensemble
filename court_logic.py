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

def verificar_votos_models(votos_models):
    # ✅ Correto
    for nome, voto in votos_models.items():
        print(f"Modelo: {nome}, Shape: {getattr(voto, 'shape', None)}")

    # ❌ INCORRETO — causaria "too many values to unpack"
    # for nome, voto in votos_models:
    #     ...

    # ✅ Também funciona, mas sem nome
    for voto in votos_models.values():
        print(f"Shape: {getattr(voto, 'shape', None)}")

def votos_models_para_lista(votos_models):
    """
    Converte um dict de votos_models para lista ordenada por nome do modelo.
    Exemplo: ['modelo_0', 'modelo_1', ..., 'modelo_n']
    Retorna: List[Tensor]
    """
    chaves_ordenadas = sorted(votos_models.keys())
    return [votos_models[k] for k in chaves_ordenadas if k in votos_models]

def safe_total_squeeze(t):
    shape = t.shape
    if shape.rank is None:
        return tf.squeeze(t)
    axes = [i for i in range(shape.rank) if shape[i] == 1]
    return tf.squeeze(t, axis=axes)

def arc_court_supreme(models, X_train, y_train, y_val, X_test, task_id=None, block_idx=None,
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

    for i, modelo in enumerate(modelos):
        try:
            _ = modelo.variables
            papel = f"JURADA_{i}" if i in [0, 1, 2] else "ADVOGADO" if i == 3 else "JUIZ" if i == 4 else "SUPREMA" if i == 5 else "PROMOTOR"
            log(f"[DEBUG] {papel} (modelo_{i}) está carregado corretamente.")
        except Exception as e:
            log(f"[ERRO] Modelo_{i} falhou na verificação de integridade: {e}")

    votos_iniciais = {}
    for i in range(5):
        log(f"[DEBUG] modelo_{i} input — shape: {X_train.shape}, min: {tf.reduce_min(X_train).numpy()}, max: {tf.reduce_max(X_train).numpy()}")
        votos_iniciais[f"modelo_{i}"] = modelos[i](X_train, training=False)

    votos_models = votos_iniciais.copy()
    votos_models = garantir_dict_votos_models(votos_models)

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

    treinar_modelo_com_y_sparse(modelos[5], X_test, y_sup, epochs=epochs * 3)

    classes_validas = extrair_classes_validas(X_test, pad_value=pad_value)
    y_antitese = inverter_classes_respeitando_valores(y_sup, classes_validas, pad_value=pad_value)
    log("[PROMOTOR] Treinando promotor com antítese da Suprema.")
    treinar_modelo_com_y_sparse(modelos[6], X_test, y_antitese, epochs=epochs)

    iter_count = 0
    consenso = 0.0

    while iter_count < max_cycles:
        skip_reeducacao = iter_count == 0
        if skip_reeducacao:
            log("[JURADOS] Modo teimoso ativado — nenhum modelo será retreinado no primeiro ciclo.")
        log(f"[CICLO] Iteração {iter_count}")

        # Atualiza os votos de todos os modelos
        for i in range(7):
            if i >= 4:
                votos_models[f"modelo_{i}"] = modelos[i](X_test, training=False)
            else:
                votos_models[f"modelo_{i}"] = modelos[i](X_train, training=False)

        # Visualização
        gerar_visualizacao_votos(
            votos_models=votos_models,
            input_tensor_outros=X_train,
            input_tensor_train=X_test,
            iteracao=iter_count,
            idx=idx,
            block_idx=block_idx,
            task_id=task_id
        )

        if iter_count > 0:
            for i in range(5):
                votos_models[f"modelo_{i}"] = modelos[i](X_train, training=False)

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

        treinar_modelo_com_y_sparse(modelos[5], X_test, y_sup, epochs=epochs)

        y_antitese = inverter_classes_respeitando_valores(y_sup, classes_validas, pad_value=pad_value)
        log("[PROMOTOR] Propondo antítese ao parecer da Suprema.")
        treinar_modelo_com_y_sparse(modelos[6], X_test, y_antitese, epochs=epochs)

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
