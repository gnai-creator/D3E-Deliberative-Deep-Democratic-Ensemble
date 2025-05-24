import tensorflow as tf
import numpy as np
from runtime_utils import log
from model_loader import load_model
from metrics_utils import salvar_voto_visual

def arc_court_supreme(models, input_tensor_outros, expected_output, block_idx=0, max_iters=10, tol=0.98, epochs=60, learning_rate=0.0005):
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
    supreme_juiza = load_model(5, learning_rate)
    MAX_CYCLES = 5

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    acc_fn = tf.keras.metrics.SparseCategoricalAccuracy()

    while consenso < 1.0:
        log(f"\n[ITER {iter_count + 1}] Iniciando rodada de julgamento")

        y_advogada_logits = advogada(input_tensor_outros, training=False)
        y_advogada_classes = tf.argmax(y_advogada_logits, axis=-1)
        log(f"[INFO] Advogada previu classes com shape: {y_advogada_classes.shape}")

        for idx, jurada in enumerate(juradas):
            jurada.fit(x=input_tensor_outros, y=y_advogada_classes, epochs=epochs, verbose=0)
            log(f"[TREINO] Jurada {idx + 1} treinada com saída da advogada")

        saidas_juradas = [jurada(input_tensor_outros, training=False) for jurada in juradas]

        def pad_to_10_channels(tensor):
            channels = tf.shape(tensor)[-1]
            padding = tf.maximum(0, 10 - channels)
            return tf.pad(tensor, paddings=[[0, 0], [0, 0], [0, 0], [0, padding]])

        juradas_padded = [pad_to_10_channels(j) for j in saidas_juradas]
        advogada_padded = pad_to_10_channels(y_advogada_logits)

        input_juiza_concat = tf.concat(juradas_padded + [advogada_padded], axis=-1)
        input_juiza_concat = tf.expand_dims(input_juiza_concat, axis=3)
        log(f"[LOG] Input juíza shape: {input_juiza_concat.shape}")

        juiza.fit(x=input_juiza_concat, y=y_advogada_classes, epochs=epochs * 3, verbose=0)
        log(f"[TREINO] Juíza treinada com opiniões de juradas e advogada")

        votos_models = [pad_to_10_channels(model(input_tensor_outros, training=False)) for model in juradas + [advogada, juiza]]

        if tf.reduce_sum(votos_models[-1]) == 0:
            log("[WARN] Juíza retornou apenas zeros na predição final.")
        else:
            log(f"[OK] Juíza produziu saída válida com shape: {votos_models[-1].shape}")

        salvar_voto_visual(votos_models, iter_count, block_idx)

        consenso = avaliar_consenso_por_j(votos_models, tol, required_votes=5)
        log(f"[CONSENSO] Iteração {iter_count + 1}: Consenso = {consenso:.4f}")

        if iter_count > 0:
            y_juiza_classes = tf.argmax(votos_models[-1], axis=-1)
            advogada.fit(x=input_tensor_outros, y=y_juiza_classes, epochs=epochs, verbose=0)
            log("[TREINO] Advogada atualizada com voto da juíza")

        log("[SUPREMA] Iniciando Suprema Juíza com julgamento do Juiz")
        loss_value = float('inf')
        accuracy = 0.0
        cycles = 0

        while (loss_value > 0.05 or accuracy < 0.68) and cycles < MAX_CYCLES:
            supreme_juiza.fit(input_tensor_outros, tf.argmax(votos_models[-1], axis=-1), epochs=epochs, verbose=0)
            pred_suprema_logits = supreme_juiza(input_tensor_outros, training=False)
            votos_supremos = tf.argmax(pred_suprema_logits, axis=-1)

            votos_supremos_logits = pad_to_10_channels(pred_suprema_logits)
            votos_models_final = [votos_supremos_logits for _ in range(6)]
            salvar_voto_visual(votos_models_final, iter_count + cycles, block_idx)

            y_true = tf.argmax(votos_models[-1], axis=-1)
            loss_value = loss_fn(y_true, pred_suprema_logits).numpy()
            acc_fn.reset_state()
            acc_fn.update_state(y_true, pred_suprema_logits)
            accuracy = acc_fn.result().numpy()

            log(f"[SUPREMA] Ciclo {cycles} - Loss: {loss_value:.4f} - Accuracy: {accuracy:.4f}")
            cycles += 1

        iter_count += 1
        votos_final = tf.argmax(votos_models[-1], axis=-1)

    log(f"\n[FIM] Julgamento encerrado após {iter_count} iteração(ões). Consenso final: {consenso:.4f}")
    return votos_supremos

def avaliar_consenso_por_j(votos_models, tol=0.98, required_votes=5):
    votos_classe = [tf.argmax(v, axis=-1) for v in votos_models]
    votos_stacked = tf.stack(votos_classe, axis=0)

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

    consenso_bin = tf.cast(votos_majoritarios >= required_votes, tf.float32)
    return tf.reduce_mean(consenso_bin).numpy()
