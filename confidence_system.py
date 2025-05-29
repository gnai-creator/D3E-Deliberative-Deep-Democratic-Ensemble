import tensorflow as tf
from runtime_utils import log
from court_utils import garantir_dict_votos_models

class ConfidenceManager:
    def __init__(self, models, initial_confidence=1.0, decay=0.9, recovery_rate=0.05, min_threshold=0.1):
        self.model_names = [f"modelo_{i}" for i in range(len(models))]
        self.confidences = {name: initial_confidence for name in self.model_names}
        self.decay = decay
        self.recovery_rate = recovery_rate
        self.min_threshold = min_threshold
        self.reintegrated = set()

    def update_confidence(self, name, acertou):
        if name not in self.confidences:
            return

        old_conf = self.confidences[name]
        if acertou:
            self.confidences[name] = min(1.0, old_conf + self.recovery_rate)
        else:
            self.confidences[name] = max(0.0, old_conf * self.decay)

    def get_confidence(self, name):
        return self.confidences.get(name, 0.0)

    def get_active_model_names(self, threshold=None):
        threshold = threshold or self.min_threshold
        ativos = [name for name, c in self.confidences.items() if c >= threshold]
        return ativos

    def log_status(self, logger=None):
        linha = "[CONFIANÇA] Modelos ativos:\n"
        for name in self.model_names:
            conf = self.get_confidence(name)
            linha += f" - {name}: {conf:.3f}\n"
        if logger:
            logger(linha)
        else:
            log(linha)


# consensus_system_weighted.py — Consenso ponderado por hierarquia de modelos
def avaliar_consenso_ponderado(votos_models: dict, pesos: dict, required_score=5.0, voto_reverso_ok=None):
    """
    Avalia consenso ponderado usando pesos hierárquicos definidos para cada modelo.
    Cada pixel é decidido com base na soma dos pesos dos modelos que concordam naquele ponto.
    """
    import tensorflow as tf
    from runtime_utils import log
    from metrics_utils import garantir_dict_votos_models

    votos_models = garantir_dict_votos_models(votos_models)
    votos_stacked = []
    votos_dict = {}

    for name, voto in votos_models.items():
        try:
            v = tf.convert_to_tensor(voto)

            if v.shape.rank >= 4 and v.shape[-1] > 1:
                v = tf.argmax(v, axis=-1)

            if v.shape.rank == 4 and v.shape[-1] == 1:
                v = tf.squeeze(v, axis=-1)

            v = tf.cast(v, tf.int64)

            if voto_reverso_ok and name in voto_reverso_ok:
                v = 9 - v

            if tf.size(v) != 900:
                log(f"[CONSENSO] ⚠️ Voto {name} tem {tf.size(v).numpy()} elementos. Ignorado.")
                continue

            votos_stacked.append(v)
            votos_dict[name] = v

        except Exception as e:
            log(f"[CONSENSO] Erro ao processar voto de {name}: {e}")

    if not votos_stacked:
        raise ValueError("Nenhum voto válido recebido para consenso.")

    votos_stacked_tensor = tf.stack(votos_stacked)  # Shape esperado: (N, 30, 30)
    altura, largura = votos_stacked_tensor.shape[1], votos_stacked_tensor.shape[2]
    resultado = tf.zeros((altura, largura), dtype=tf.int64)

    for i in range(altura):
        for j in range(largura):
            valores_pixel = tf.reshape(votos_stacked_tensor[:, i, j], [-1])  # Garante 1D
            valores, _, counts = tf.unique_with_counts(valores_pixel)
            scores = []

            for idx, val in enumerate(valores):
                classe = val.numpy()
                peso_total = 0.0
                for model_name, voto_tensor in votos_dict.items():
                    if voto_tensor[i, j].numpy().item() == classe:
                        peso_total += pesos.get(model_name, 1.0)

                scores.append((classe, peso_total))

            if scores:
                melhor_classe, melhor_score = max(scores, key=lambda x: x[1])
                if melhor_score >= required_score:
                    resultado = tf.tensor_scatter_nd_update(
                        resultado,
                        indices=[[i, j]],
                        updates=[melhor_classe]
                    )

    return tf.expand_dims(tf.expand_dims(resultado, axis=0), axis=-1)  # (1, 30, 30, 1)
