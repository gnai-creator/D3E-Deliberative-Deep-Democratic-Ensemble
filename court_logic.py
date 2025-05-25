import tensorflow as tf
import numpy as np
import os
from runtime_utils import log
from model_loader import load_model
from metrics_utils import salvar_voto_visual
from confidence_system import ConfidenceManager, avaliar_consenso_com_confiança

LOGFILE = "logs/deliberacao.log"
os.makedirs("logs", exist_ok=True)

def logf(msg):
    print(msg)
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")

def arc_court_supreme(models, input_tensor_outros, task_id=None, block_idx=0,
                      max_iters=10, tol=0.98, epochs=60, learning_rate=0.0005):

    if len(models) < 5:
        raise ValueError("Corte incompleta: recebi menos de 5 modelos.")

    juradas = models[:3]
    advogada = models[3]
    juiza = models[4]
    supreme_juiza = load_model(5, learning_rate)

    model_dict = {
        "jurada_1": juradas[0],
        "jurada_2": juradas[1],
        "jurada_3": juradas[2],
        "advogada": advogada,
        "juiza": juiza,
        "suprema": supreme_juiza,
    }

    manager = ConfidenceManager(model_dict)
    iter_count = 0
    consenso = 0.0
    MAX_CYCLES = 50
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_fn = tf.keras.metrics.SparseCategoricalAccuracy()

    while consenso < 1.0 and iter_count < max_iters:
        logf(f"\n[ITER {iter_count + 1}] Iniciando rodada de julgamento")

        if len(input_tensor_outros.shape) == 4 and input_tensor_outros.shape[0] != 1:
            input_tensor_outros = tf.expand_dims(input_tensor_outros, axis=0)

        # ADVOGADA
        y_advogada_logits = advogada(input_tensor_outros, training=False)
        y_advogada_classes = tf.argmax(y_advogada_logits, axis=-1)

        logf(f"[INFO] Advogada previu classes. Shape dos logits: {y_advogada_logits.shape}")

        # JURADAS
        for idx, jurada in enumerate(juradas):
            jurada.fit(input_tensor_outros, y_advogada_classes, epochs=epochs, verbose=0)
            logf(f"[TREINO] Jurada_{idx + 1} treinada com saída da advogada")

        saidas_juradas = [jurada(input_tensor_outros, training=False) for jurada in juradas]

        # JUÍZA
        def pad_to_10_channels(tensor):
            ch = tf.shape(tensor)[-1]
            padding = tf.maximum(0, 10 - ch)
            return tf.pad(tensor, paddings=[[0,0],[0,0],[0,0],[0,padding]])

        juradas_padded = [pad_to_10_channels(s) for s in saidas_juradas]
        advogada_padded = pad_to_10_channels(y_advogada_logits)
        input_juiza = tf.concat(juradas_padded + [advogada_padded], axis=-1)
        input_juiza = tf.expand_dims(input_juiza, axis=3)
        juiza.fit(input_juiza, y_advogada_classes, epochs=epochs * 3, verbose=0)
        logf("[TREINO] Juíza treinada com opiniões")

        # VOTAÇÃO
        votos_models = {}
        for idx, nome in enumerate(["jurada_1", "jurada_2", "jurada_3", "advogada", "juiza"]):
            model = model_dict[nome]
            if hasattr(model, "from_40"):
                input_tensor_mod = tf.concat(list(votos_models.values()), axis=-1)
                input_tensor_mod = input_tensor_mod[..., :40]
                input_tensor_mod = tf.expand_dims(input_tensor_mod, axis=3)
            else:
                input_tensor_mod = input_tensor_outros
            pred = model(input_tensor_mod, training=False)
            padded = pad_to_10_channels(pred)
            votos_models[nome] = padded
            logf(f"[VOTO] {nome} - shape: {padded.shape}")

        # SUPREMA
        logf("[SUPREMA] Iniciando Suprema Juíza com julgamento")
        entrada_suprema = tf.concat(list(votos_models.values()), axis=-1)
        entrada_suprema = entrada_suprema[..., :40]
        entrada_suprema = tf.reshape(entrada_suprema, [1, 30, 30, 40])
        entrada_suprema = tf.expand_dims(entrada_suprema, axis=3)

        loss_value, accuracy, cycles = float("inf"), 0.0, 0
        while (loss_value > 0.05 or accuracy < 0.95) and cycles < MAX_CYCLES:
            supreme_juiza.fit(entrada_suprema, y_advogada_classes, epochs=epochs, verbose=0)
            pred_suprema = supreme_juiza(entrada_suprema, training=False)
            y_pred_suprema = tf.argmax(pred_suprema, axis=-1)
            loss_value = loss_fn(y_advogada_classes, pred_suprema).numpy()
            acc_fn.reset_state()
            acc_fn.update_state(y_advogada_classes, pred_suprema)
            accuracy = acc_fn.result().numpy()
            logf(f"[SUPREMA] Ciclo {cycles} - Loss: {loss_value:.4f} - Acc: {accuracy:.4f}")
            cycles += 1

        votos_models["suprema"] = pad_to_10_channels(pred_suprema)

        salvar_voto_visual(list(votos_models.values()), iter_count, block_idx, input_tensor_outros, task_id=task_id)

        manager.update_confidences(votos_models, y_pred_suprema)
        manager.reabilitar_modelos()

        consenso = avaliar_consenso_com_confiança(
            votos_models,
            confidence_manager=manager,
            required_votes=5,
            confidence_threshold=0.5
        )

        logf(f"[CONSENSO] Iteração {iter_count+1}: {consenso:.4f}")
        iter_count += 1

    logf(f"\n[FIM] Julgamento encerrado após {iter_count} iteração(ões). Consenso final: {consenso:.4f}")
    return y_pred_suprema
