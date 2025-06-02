# court_logic.py — Versão Corrigida para Pipeline de Classes Discretas
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
from court_utils import extrair_classe_cor, expandir_para_3_canais

tf.config.run_functions_eagerly(True)

pesos = {
    "modelo_0": 1.2,
    "modelo_1": 1.5,
    "modelo_2": 2.0,
    "modelo_3": -1.7,
}
def robust_int_label(tensor, n_classes=10):
    """
    Converte o tensor para int32 após arredondar corretamente e limitar as classes válidas.
    Espera shape (batch, 30, 30, [1]) ou (batch, 30, 30).
    """
    tensor_float = tf.cast(tensor, tf.float32)
    rounded = tf.floor(tensor_float + 0.5)                 # Arredondamento correto
    clipped = tf.clip_by_value(rounded, 0, n_classes-1)    # Limita ao range válido
    return tf.cast(clipped, tf.int32)

def mutar_label(y, p=0.03):
    y_np = y.numpy().copy()
    mask = np.random.rand(*y_np.shape) < p
    classes = np.unique(y_np)
    if len(classes) > 1:
        for c in classes:
            idxs = np.where(mask & (y_np == c))
            outras = [v for v in classes if v != c]
            if outras:
                y_np[idxs] = np.random.choice(outras, size=len(idxs[0]))
    return tf.convert_to_tensor(y_np)

def arc_court_supreme(models, X_train, y_train, y_val, X_test, task_id=None, block_idx=None,
                      max_cycles=20, tol=9.0, epochs=10, confidence_threshold=0.5,
                      confidence_manager=[], idx=0, pad_value=-1, Y_val=None):
    log(f"[SUPREMA] Iniciando deliberação para o bloco {block_idx} — task {task_id}")

    modelos = models.copy()

    classes_validas = extrair_todas_classes_validas(X_test, X_train, pad_value=pad_value)
    classes_objetivo = extrair_classes_validas(X_test, pad_value=pad_value)

    # Inicialização dos votos
    votos_models = {
        "modelo_0": modelos[0](X_test, training=False),
        "modelo_1": modelos[1](X_train, training=False),
        "modelo_2": modelos[2](X_test, training=False),
        "modelo_3": modelos[3](X_test, training=False),
    }
    # Treina Suprema e Promotor já na largada
    preds_stack = tf.stack(
        [tf.squeeze(extrair_classe_cor(votos_models[f"modelo_{i}"]), axis=0) for i in [0, 1]],
        axis=0
    )  # shape (2, 30, 30)
    y_sup = pixelwise_mode(preds_stack, pad_value=pad_value)
    y_sup_recolorido = mapear_cores_para_x_test(y_sup, classes_objetivo)
    y_antitese = inverter_classes_respeitando_valores(y_sup_recolorido, classes_objetivo, pad_value=pad_value)
    y_sup_redi = expandir_para_3_canais(y_sup_recolorido)
    y_antitese_redi = expandir_para_3_canais(y_antitese)
    
    # ... Já treinou modelo_1 (Advogada) com X_train/y_train
    for i in range (2,4):
        weights_path = f"weights_model_{i}_block_{block_idx}.h5"
        if os.path.exists(weights_path):
            modelos[i].load_weights(weights_path)
            log(f"[INFO] Pesos carregados: {weights_path}")
        else:
            log(f"[AVISO] Pesos não encontrados: {weights_path} — pulando carregamento.")

    iter_count = 0
    consenso = 0.0
    match = 0.0
    threshold_pixel = 0.7
    threshold_shape = 0.7
  
    # Warmup dinâmico para Suprema e Promotor com ajuste de shape e tipagem correta
    warmup = 0
    pixel_perfect_sup = 0.0
    shape_perfect_sup = 0.0
    pixel_perfect_pro = 0.0
    shape_perfect_pro = 0.0

    while not (pixel_perfect_sup > threshold_pixel and shape_perfect_sup > threshold_shape and
                pixel_perfect_pro > threshold_pixel and shape_perfect_pro > threshold_shape) :
        # Advogada prediz pseudo-label no X_test
        pseudo_labels_sup = modelos[1](X_test, training=False)
        pseudo_labels_sup = extrair_classe_cor(pseudo_labels_sup)
        if len(pseudo_labels_sup.shape) == 3:
            pseudo_labels_sup = tf.expand_dims(pseudo_labels_sup, -1)  # (1, 30, 30, 1)
        pseudo_labels_sup_rgb = expandir_para_3_canais(pseudo_labels_sup)

        # Treina Suprema e Promotor
        treinar_modelo_com_y_sparse(modelos[2], X_test, pseudo_labels_sup_rgb,model_idx=2, idx=block_idx, epochs=epochs*6)

        pseudo_labels_antitese = inverter_classes_respeitando_valores(pseudo_labels_sup, classes_objetivo, pad_value=pad_value)
        if len(pseudo_labels_antitese.shape) == 3:
            pseudo_labels_antitese = tf.expand_dims(pseudo_labels_antitese, -1)
        pseudo_labels_antitese_rgb = expandir_para_3_canais(pseudo_labels_antitese)
        treinar_modelo_com_y_sparse(modelos[3], X_test, pseudo_labels_antitese_rgb, model_idx=3, idx=block_idx, epochs=epochs*6)

        # Verifica pixel-perfect
        pred_suprema = extrair_classe_cor(modelos[2](X_test, training=False))
        pred_promotor = extrair_classe_cor(modelos[3](X_test, training=False))

        # Cast para int32 antes de comparar!
        pred_suprema = robust_int_label(pred_suprema)
        pred_promotor = robust_int_label(pred_promotor)
        pseudo_labels_sup = robust_int_label(pseudo_labels_sup)
        pseudo_labels_antitese = robust_int_label(pseudo_labels_antitese)

        pixel_perfect_sup = tf.reduce_mean(tf.cast(tf.equal(pred_suprema, pseudo_labels_sup), tf.float32)).numpy()
        pixel_perfect_pro = tf.reduce_mean(tf.cast(tf.equal(pred_promotor, pseudo_labels_antitese), tf.float32)).numpy()


        classes_adv = np.unique(pseudo_labels_sup.numpy())
        classes_sup = np.unique(pred_suprema.numpy())
        shape_perfect_sup = len(np.intersect1d(classes_adv, classes_sup)) / max(1, len(classes_adv))

        classes_ant = np.unique(pseudo_labels_antitese.numpy())
        classes_pro = np.unique(pred_promotor.numpy())
        shape_perfect_pro = len(np.intersect1d(classes_ant, classes_pro)) / max(1, len(classes_ant))

        votos_models = {
            "modelo_0": modelos[0](X_test, training=False),
            "modelo_1": modelos[1](X_train, training=False),
            "modelo_2": modelos[2](X_test, training=False),
            "modelo_3": modelos[3](X_test, training=False),
        }

        gerar_visualizacao_votos(
            votos_models=votos_models,
            input_tensor_outros=X_train,
            input_tensor_train=X_test,
            iteracao=warmup,  # Valor negativo para distinguir fase de warmup
            idx=idx,
            block_idx=block_idx,
            task_id=task_id,
            classes_validas=classes_validas,
            classes_objetivo=classes_objetivo,
            consenso=consenso,
            fase='Warmup'
        )

        print(f'[WARMUP {warmup}] Suprema: pixel={pixel_perfect_sup:.2f}, shape={shape_perfect_sup:.2f} | Promotor: pixel={pixel_perfect_pro:.2f}, shape={shape_perfect_pro:.2f}')
        warmup += 1


    modelos[2].save_weights(f"weights_model_{2}_block_{block_idx}.h5")
    modelos[3].save_weights(f"weights_model_{3}_block_{block_idx}.h5")
    while consenso <= tol : #and iter_count < max_cycles:
        log(f"[DEBUG] iter_count={iter_count}, block_idx={block_idx}, idx={idx}, task_id={task_id}")

        # Atualiza votos: modelo_0 e modelo_1 usam X_test/X_train
        votos_models["modelo_0"] = modelos[0](X_test, training=False)
        votos_models["modelo_1"] = modelos[1](X_train, training=False)
        votos_models["modelo_2"] = modelos[2](X_test, training=False)
        votos_models["modelo_3"] = modelos[3](X_test, training=False)
        votos_models = garantir_dict_votos_models(votos_models)

        # Log das previsões
        for k, p in votos_models.items():
            if isinstance(p, tf.Tensor):
                pred = tf.squeeze(extrair_classe_cor(p), axis=0)
                log(f"[DEBUG] {k} unique: {np.unique(pred.numpy())} shape: {pred.shape}")

        # Stack para consenso
        # Durante as primeiras iterações (ou até um critério de diversidade ser atingido), só inclua 0 e 1:
        modelos_consenso = [0,1] if match < 0.6 else [0, 1, 2, 3]
        preds_stack = tf.stack(
            [tf.squeeze(extrair_classe_cor(votos_models[f"modelo_{i}"]), axis=0) for i in modelos_consenso],
            axis=0
        )


        log(f"[DEBUG] preds_stack shape: {preds_stack.shape}")
        log(f"[DEBUG] preds_stack unique: {np.unique(preds_stack.numpy())}")

        y_sup = pixelwise_mode(preds_stack, pad_value=pad_value)
        y_sup_recolorido = mapear_cores_para_x_test(y_sup, classes_objetivo)
        y_antitese = inverter_classes_respeitando_valores(y_sup_recolorido, classes_objetivo, pad_value=pad_value)
        log(f"[DEBUG] classes_validas: {classes_validas}")
        log(f"[DEBUG] classes_objetivo: {classes_objetivo}")
        log(f"[DEBUG] pixelwise_mode unique: {np.unique(y_sup)}")
        log(f"[DEBUG] y_sup_recolorido unique: {np.unique(y_sup_recolorido)}")
        log(f"[DEBUG] y_antitese unique: {np.unique(y_antitese)}")

        y_sup_redi = expandir_para_3_canais(y_sup_recolorido)
        y_antitese_redi = expandir_para_3_canais(mutar_label(y_antitese, p=0.15))

        # Treina Suprema e Promotor
        treinar_modelo_com_y_sparse(modelos[2], X_test, y_sup_redi, model_idx=2, idx=block_idx, epochs=epochs * 3)
        treinar_modelo_com_y_sparse(modelos[3], X_test, y_antitese_redi, model_idx=3, idx=block_idx, epochs=epochs * 3)

        # Match entre Suprema e consenso dos outros
        if iter_count >= max_cycles/4:
            preds_stack_others = tf.stack(
                [tf.squeeze(extrair_classe_cor(p), axis=0) for k, p in votos_models.items() if k != "modelo_0"], axis=0
            )
            y_sup_others = pixelwise_mode(preds_stack_others, pad_value=pad_value)
            y_pred = tf.squeeze(extrair_classe_cor(modelos[0](X_test, training=False)), axis=0)
            y_target = y_sup_others

            # CORREÇÃO: cast ambos para int32!
            y_pred = tf.cast(y_pred, tf.int32)
            y_target = tf.cast(y_target, tf.int32)

            log(f"[DEBUG] y_pred unique: {np.unique(y_pred.numpy())}")
            log(f"[DEBUG] y_target unique: {np.unique(y_target.numpy())}")
            match = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_target), tf.float32)).numpy()
            log(f"[MATCH] MATCH {match}")

            if iter_count % 3 == 0:
                treinar_modelo_com_y_sparse(modelos[0], X_test, y_sup_redi, model_idx=0, idx=block_idx, epochs=epochs * 3)
            elif match <= 0.97:
                treinar_modelo_com_y_sparse(modelos[0], X_test, y_antitese_redi, model_idx=0, idx=block_idx, epochs=epochs * 3)

        votos_models = garantir_dict_votos_models(votos_models)
        resultado, consenso = avaliar_consenso_ponderado(
            votos_models,
            pesos,
            required_score=5.0,
            voto_reverso_ok=["modelo_3"]
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
            consenso=consenso,
            fase='Test'
        )
        log(f"[CONSENSO] : CONSENSO {consenso}")
        if consenso >= tol:
            return {
                "class_logits": votos_models["modelo_2"],
                "consenso": float(consenso),
                "y_pred_simbolico": y_sup
            }
        iter_count += 1

    for model_idx in range(len(models)):
        models[model_idx].save_weights(f"weights_model_{model_idx}_block_{block_idx}.h5")

    return {
        "class_logits": votos_models["modelo_2"],
        "consenso": float(consenso),
        "y_pred_simbolico": y_sup
    }
