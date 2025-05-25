import tensorflow as tf
import numpy as np
from runtime_utils import log
from model_loader import load_model
from metrics_utils import salvar_voto_visual
from confidence_system import init_confidence, update_confidence, get_valid_voters, restore_confidence

def arc_court_supreme(models, input_tensor_outros, expected_output, task_id, block_idx=0, max_iters=10, tol=0.78, epochs=160, learning_rate=0.0005):
    if len(models) < 5:
        raise ValueError("Corte incompleta: recebi menos de 5 modelos.")

    juradas = [models[i] for i in range(3)]
    advogada = models[3]
    juiza = models[4]
    nomes_modelos = ["Jurada 1", "Jurada 2", "Jurada 3", "Advogada", "Juíza"]

    consenso = 0.0
    iter_count = 0
    votos_final = None

    confidence = init_confidence(n=len(models))
    supreme_juiza = load_model(5, learning_rate)
    MAX_CYCLES = 150

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_fn = tf.keras.metrics.SparseCategoricalAccuracy()

    while consenso < 1.0 and iter_count < max_iters:
        if len(input_tensor_outros.shape) == 4 and input_tensor_outros.shape[0] != 1:
            input_tensor_outros = tf.expand_dims(input_tensor_outros, axis=0)
        log(f"\n[ITER {iter_count + 1}] Julgamento iniciado. Confianças: {confidence}")

        y_advogada_logits = advogada(input_tensor_outros, training=False)
        y_advogada_classes = tf.argmax(y_advogada_logits, axis=-1)

        for idx, jurada in enumerate(juradas):
            if confidence[idx] > 0:
                if idx in [0, 1]:  # juradas do contra
                    y_treino = (tf.cast(y_advogada_classes, tf.int64) +
                                tf.random.uniform(shape=y_advogada_classes.shape, maxval=3, dtype=tf.int64)) % 10
                else:
                    y_treino = y_advogada_classes
                jurada.fit(x=input_tensor_outros, y=y_treino, epochs=epochs, verbose=0)
                log(f"[TREINO] {nomes_modelos[idx]} treinada.")

        saidas_juradas = [juradas[i](input_tensor_outros, training=False) if confidence[i] > 0 else tf.zeros_like(y_advogada_logits) for i in range(3)]
        juradas_padded = [pad_to_10_channels(j) for j in saidas_juradas]
        advogada_padded = pad_to_10_channels(y_advogada_logits)

        input_juiza = tf.concat(juradas_padded + [advogada_padded], axis=-1)
        input_juiza = tf.expand_dims(input_juiza, axis=3)

        juiza.fit(x=input_juiza, y=y_advogada_classes, epochs=epochs * 3, verbose=0)

        votos_models = []
        modelos = juradas + [advogada, juiza]
        for idx, model in enumerate(modelos):
            if confidence[idx] > 0:
                if hasattr(model, 'from_40'):
                    input_tensor_mod = tf.concat(votos_models, axis=-1)[..., :40]
                    input_tensor_mod = tf.expand_dims(input_tensor_mod, axis=3)
                else:
                    input_tensor_mod = input_tensor_outros
                pred = model(input_tensor_mod, training=False)
                padded = pad_to_10_channels(pred)
            else:
                padded = tf.zeros_like(advogada_padded)
            votos_models.append(padded)

        consenso = avaliar_consenso_por_j(votos_models, tol, required_votes=5)
        log(f"[CONSENSO] Iteração {iter_count + 1}: {consenso:.4f}")

        # Suprema Juíza
        entrada_suprema = input_tensor_outros
        if entrada_suprema.shape[-1] < 40:
            padding = 40 - entrada_suprema.shape[-1]
            entrada_suprema = tf.pad(entrada_suprema, [[0, 0], [0, 0], [0, 0], [0, 0], [0, padding]])
        entrada_suprema = tf.reshape(entrada_suprema, [1, 30, 30, 1, 40])

        loss_value = float('inf')
        accuracy = 0.0
        cycles = 0
        while (loss_value > 0.01 or accuracy < 0.98) and cycles < MAX_CYCLES:
            supreme_juiza.fit(entrada_suprema, tf.argmax(votos_models[-1], axis=-1), epochs=epochs, verbose=0)
            pred_suprema_logits = supreme_juiza(entrada_suprema, training=False)
            votos_supremos = tf.argmax(pred_suprema_logits, axis=-1)

            y_true = tf.argmax(votos_models[-1], axis=-1)
            loss_value = loss_fn(y_true, pred_suprema_logits).numpy()
            acc_fn.reset_state()
            acc_fn.update_state(y_true, pred_suprema_logits)
            accuracy = acc_fn.result().numpy()

            votos_supremos_logits = pad_to_10_channels(pred_suprema_logits)
            salvar_voto_visual([votos_supremos_logits]*6, iter_count + cycles, block_idx, input_tensor_outros, task_id)
            log(f"[SUPREMA] Ciclo {cycles} - Loss: {loss_value:.4f} - Accuracy: {accuracy:.4f}")
            cycles += 1

        # Advogada aprende com Suprema
        adv_input = tf.reshape(entrada_suprema[..., :4], [1, 30, 30, 1, 4])
        advogada.fit(x=adv_input, y=votos_supremos, epochs=epochs, verbose=0)

        # Atualiza confiança
        confidence = update_confidence(confidence, votos_models, votos_supremos)
        confidence = restore_confidence(confidence, votos_models, votos_supremos)

        iter_count += 1
        votos_final = votos_supremos

    log(f"\n[FIM] Julgamento finalizado em {iter_count} iterações. Consenso: {consenso:.4f}")
    return votos_final

def pad_to_10_channels(tensor):
    channels = tf.shape(tensor)[-1]
    padding = tf.maximum(0, 10 - channels)
    return tf.pad(tensor, paddings=[[0, 0], [0, 0], [0, 0], [0, padding]])

def avaliar_consenso_por_j(votos_models, tol=0.98, required_votes=5):
    votos_classe = [tf.argmax(v, axis=-1) for v in votos_models]
    votos_stacked = tf.stack(votos_classe, axis=0)

    def contar_consenso(votos_pixel):
        _, _, count = tf.unique_with_counts(votos_pixel)
        return tf.reduce_max(count)

    votos_majoritarios = tf.map_fn(
        lambda x: tf.map_fn(lambda y: tf.map_fn(contar_consenso, y, dtype=tf.int32), x, dtype=tf.int32),
        tf.transpose(votos_stacked, [1, 2, 3, 0]),
        dtype=tf.int32
    )

    consenso_bin = tf.cast(votos_majoritarios >= required_votes, tf.float32)
    return tf.reduce_mean(consenso_bin).numpy()
