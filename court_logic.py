import tensorflow as tf
import numpy as np
from runtime_utils import log
from model_loader import load_model
from metrics_utils import salvar_voto_visual
from confidence_system import ConfidenceManager, avaliar_consenso_com_confiança


def arc_court_supreme(models, input_tensor_outros, task_id=None, block_idx=0, max_iters=10, tol=0.98, epochs=60, learning_rate=0.0005):
    if len(models) < 5:
        raise ValueError("Corte incompleta: recebi menos de 5 modelos.")

    nomes = ["jurada_1", "jurada_2", "jurada_3", "advogada", "juiza"]
    juradas = dict(zip(nomes[:3], models[:3]))
    advogada = models[3]
    juiza = models[4]

    model_dict = dict(zip(nomes, models))
    manager = ConfidenceManager(model_dict)

    supreme_juiza = load_model(5, learning_rate)

    iter_count = 0
    consenso = 0.0
    MAX_CYCLES = 150

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_fn = tf.keras.metrics.SparseCategoricalAccuracy()

    def pad_to_10_channels(tensor):
        diff = 10 - tensor.shape[-1]
        if diff > 0:
            return tf.pad(tensor, [[0, 0], [0, 0], [0, 0], [0, diff]])
        return tensor

    while consenso < 1.0 and iter_count < max_iters:
        if len(input_tensor_outros.shape) == 4:
            input_tensor_outros = tf.expand_dims(input_tensor_outros, axis=0)

        log(f"[ITER {iter_count}] Início do julgamento")

        y_advogada_logits = advogada(input_tensor_outros, training=False)
        y_advogada_classes = tf.argmax(y_advogada_logits, axis=-1)

        for name, jurada in juradas.items():
            jurada.fit(input_tensor_outros, y_advogada_classes, epochs=epochs, verbose=0)

        saidas_juradas = {name: jurada(input_tensor_outros, training=False) for name, jurada in juradas.items()}
        juradas_padded = {name: pad_to_10_channels(logits) for name, logits in saidas_juradas.items()}
        advogada_padded = pad_to_10_channels(y_advogada_logits)

        input_juiza_concat = tf.concat(list(juradas_padded.values()) + [advogada_padded], axis=-1)
        input_juiza_concat = tf.expand_dims(input_juiza_concat, axis=3)

        juiza.fit(input_juiza_concat, y_advogada_classes, epochs=epochs * 2, verbose=0)

        votos_models = {}
        for name, model in model_dict.items():
            if hasattr(model, "from_40"):
                entrada = tf.concat(list(juradas_padded.values()) + [advogada_padded], axis=-1)[..., :40]
                entrada = tf.expand_dims(entrada, axis=3)
            else:
                entrada = input_tensor_outros
            pred = model(entrada, training=False)
            votos_models[name] = pad_to_10_channels(pred)

        if tf.reduce_sum(votos_models["juiza"]) == 0:
            log("[WARN] Juíza retornou apenas zeros na predição final.")

        salvar_voto_visual(list(votos_models.values()), iter_count, block_idx, input_tensor_outros, task_id=task_id)

        consenso = avaliar_consenso_com_confiança(
            votos_models, confidence_manager=manager, required_votes=5, confidence_threshold=0.5
        )
        log(f"[CONSENSO] Iteração {iter_count}: Consenso = {consenso:.4f}")

        entrada_suprema = tf.concat(list(votos_models.values()), axis=-1)
        if entrada_suprema.shape[-1] >= 40:
            entrada_suprema = entrada_suprema[..., :40]
        else:
            entrada_suprema = tf.pad(entrada_suprema, [[0, 0], [0, 0], [0, 0], [0, 40 - entrada_suprema.shape[-1]]])
        entrada_suprema = tf.reshape(entrada_suprema, [1, 30, 30, 40])
        entrada_suprema = tf.expand_dims(entrada_suprema, axis=3)

        loss_value = float('inf')
        accuracy = 0.0
        cycles = 0

        while (loss_value > 0.05 or accuracy < 0.95) and cycles < MAX_CYCLES:
            supreme_juiza.fit(entrada_suprema, tf.argmax(votos_models["juiza"], axis=-1), epochs=epochs, verbose=0)
            pred_suprema_logits = supreme_juiza(entrada_suprema, training=False)
            votos_supremos = tf.argmax(pred_suprema_logits, axis=-1)

            y_true = tf.argmax(votos_models["juiza"], axis=-1)
            loss_value = loss_fn(y_true, pred_suprema_logits).numpy()
            acc_fn.reset_state()
            acc_fn.update_state(y_true, pred_suprema_logits)
            accuracy = acc_fn.result().numpy()

            votos_supremos_logits = pad_to_10_channels(pred_suprema_logits)
            votos_models_final = [votos_supremos_logits for _ in range(6)]
            salvar_voto_visual(votos_models_final, iter_count + cycles, block_idx, input_tensor_outros, task_id=task_id)

            log(f"[SUPREMA] Ciclo {cycles} - Loss: {loss_value:.4f} - Accuracy: {accuracy:.4f}")
            cycles += 1

        input_advogada = tf.concat(list(votos_models.values()), axis=-1)[..., :4]
        input_advogada = tf.reshape(input_advogada, [1, 30, 30, 1, 4])
        advogada.fit(input_advogada, votos_supremos, epochs=epochs, verbose=0)

        manager.update_confidences(votos_models, votos_supremos)
        iter_count += 1

    log(f"[FIM] Julgamento finalizado após {iter_count} iteração(ões). Consenso: {consenso:.4f}")
    return votos_supremos

