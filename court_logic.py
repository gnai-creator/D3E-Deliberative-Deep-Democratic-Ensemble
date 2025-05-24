import tensorflow as tf
import numpy as np
from runtime_utils import log
from metrics_utils import salvar_voto_visual


def arc_court(models, input_tensor_outros, max_iters=10, tol=0.98, epochs=60):
    if len(models) < 5:
        raise ValueError("Corte incompleta: recebi menos de 5 modelos.")

    juradas = [models[i] for i in range(3)]
    advogada = models[3]
    juiza = models[4]

    consenso = 0.0
    iter_count = 0
    votos_final = None

    log(f"[INÍCIO] Tribunal iniciado com {len(models)} modelos.")
    log(f"[INFO] Tolerância de consenso definida em {tol:.2f}")

    while consenso < 1.0:
        log(f"\n[ITER {iter_count + 1}] Iniciando rodada de julgamento")

        # 1. Advogada faz predição
        y_advogada_logits = advogada(input_tensor_outros, training=False)
        y_advogada_classes = tf.argmax(y_advogada_logits, axis=-1)
        log(f"[INFO] Advogada previu classes com shape: {y_advogada_classes.shape}")

        # 2. Juradas aprendem com a advogada
        for idx, jurada in enumerate(juradas):
            jurada.fit(x=input_tensor_outros, y=y_advogada_classes, epochs=epochs, verbose=0)
            log(f"[TREINO] Jurada {idx + 1} treinada com saída da advogada")

        # 3. Juradas produzem predições
        saidas_juradas = [jurada(input_tensor_outros, training=False) for jurada in juradas]

        # 4. Padroniza todos os tensores para 10 canais
        def pad_to_10_channels(tensor):
            channels = tf.shape(tensor)[-1]
            padding = tf.maximum(0, 10 - channels)
            return tf.pad(tensor, paddings=[[0, 0], [0, 0], [0, 0], [0, padding]])

        juradas_padded = [pad_to_10_channels(j) for j in saidas_juradas]
        advogada_padded = pad_to_10_channels(y_advogada_logits)

        # 5. Juíza aprende com concatenação dos canais (mantém 40)
        input_juiza_concat = tf.concat(juradas_padded + [advogada_padded], axis=-1)
        input_juiza_concat = tf.expand_dims(input_juiza_concat, axis=3)
        log(f"[LOG] Input juíza shape: {input_juiza_concat.shape}")

        juiza.fit(x=input_juiza_concat, y=y_advogada_classes, epochs=epochs * 3, verbose=0)
        log(f"[TREINO] Juíza treinada com opiniões de juradas e advogada")

        # 6. Todos votam
        votos_models = [pad_to_10_channels(model(input_tensor_outros, training=False)) for model in juradas + [advogada]]
        entrada_juiza_final = tf.concat(votos_models, axis=-1)
        entrada_juiza_final = tf.expand_dims(entrada_juiza_final, axis=3)
        voto_juiza = juiza(entrada_juiza_final, training=False)

        # Diagnóstico
        # log(f"[DEBUG] Juiza output preview: {tf.reduce_mean(voto_juiza)} {tf.reduce_max(voto_juiza)} {tf.shape(voto_juiza)}")
        # y_pred_classes = tf.argmax(voto_juiza, axis=-1)
        # log(f"[DEBUG] JUIZA PRED CLASSES: {tf.unique(tf.reshape(y_pred_classes, [-1]))}")

        if tf.reduce_sum(voto_juiza) == 0:
            log("[WARN] Juíza retornou apenas zeros na predição final.")
        else:
            log(f"[OK] Juíza produziu saída válida com shape: {voto_juiza.shape}")

        votos_models.append(voto_juiza)

        # 7. Plotar votos
        salvar_voto_visual(votos_models, iter_count)

        # 8. Avaliar consenso
        consenso = avaliar_consenso_por_j(votos_models, tol)
        log(f"[CONSENSO] Iteração {iter_count + 1}: Consenso = {consenso:.4f}")

        # 9. Treinar advogada com saída da juíza
        if iter_count > 0:
            y_juiza_classes = tf.argmax(voto_juiza, axis=-1)
            advogada.fit(x=input_tensor_outros, y=y_juiza_classes, epochs=epochs, verbose=0)
            log("[TREINO] Advogada atualizada com voto da juíza")

        iter_count += 1
        votos_final = tf.argmax(voto_juiza, axis=-1)

    log(f"\n[FIM] Julgamento encerrado após {iter_count} iteração(ões). Consenso final: {consenso:.4f}")
    return votos_final


def avaliar_consenso_por_j(votos_models, tol=0.98):
    votos_classe = [tf.argmax(v, axis=-1) for v in votos_models]
    votos_stacked = tf.stack(votos_classe, axis=0)  # [N, B, H, W]

    def contar_consenso(votos_pixel):
        uniques, _, count = tf.unique_with_counts(votos_pixel)
        return tf.reduce_max(count)

    votos_majoritarios = tf.map_fn(
        lambda x: tf.map_fn(
            lambda y: tf.map_fn(contar_consenso, y, dtype=tf.int32),
            x,
            dtype=tf.int32
        ),
        tf.transpose(votos_stacked, [1, 2, 3, 0]),
        dtype=tf.int32
    )

    consenso_bin = tf.cast(votos_majoritarios >= 3, tf.float32)
    return tf.reduce_mean(consenso_bin).numpy()