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
    Aplica veto epistêmico apenas se a Suprema divergir significativamente da maioria (>5% dos pixels).
    """
    votos_stacked = []
    votos_dict = {}
    pesos_usados = []

    for name, voto in garantir_dict_votos_models(votos_models).items():
        try:
            v = tf.convert_to_tensor(voto)
            log(f"[CONSENSO] DEBUG: {name} - shape inicial: {v.shape}, rank: {v.shape.rank}")

            if v.shape.rank >= 4 and v.shape[-1] > 1:
                v = tf.argmax(v, axis=-1)
                log(f"[CONSENSO] DEBUG: {name} - shape após argmax: {v.shape}")
            if v.shape.rank == 4 and v.shape[-1] == 1:
                v = tf.squeeze(v, axis=-1)
                log(f"[CONSENSO] DEBUG: {name} - shape após squeeze[-1]: {v.shape}")
            if v.shape.rank == 4 and v.shape[0] == 1:
                v = tf.squeeze(v, axis=0)
                log(f"[CONSENSO] DEBUG: {name} - shape após squeeze[0]: {v.shape}")

            v = tf.cast(v, tf.int64)

            if voto_reverso_ok and name in voto_reverso_ok:
                v = 9 - v

            if tf.size(v) != 900 and v.shape != (30, 30, 1):
                log(f"[CONSENSO] ⚠️ Voto {name} tem shape inesperado {v.shape}. Ignorado.")
                continue

            votos_stacked.append(v)
            votos_dict[name] = v
            pesos_usados.append(pesos.get(name, 1.0))

        except Exception as e:
            log(f"[CONSENSO] Erro ao processar voto de {name}: {e}")

    if not votos_stacked:
        log("[CONSENSO] Nenhum voto válido após filtragem.")
        return 0.0

    if "modelo_5" in votos_dict:
        voto_suprema = tf.cast(votos_dict["modelo_5"], tf.int64)
        votos_sem_suprema = [tf.cast(v, tf.int64) for k, v in votos_dict.items() if k != "modelo_5" and k != "modelo_6"]
        if votos_sem_suprema:
            votos_moda = tf.cast(tf.round(tf.reduce_mean(tf.stack(votos_sem_suprema), axis=0)), tf.int64)
            divergencia_ratio = tf.reduce_mean(tf.cast(tf.not_equal(voto_suprema, votos_moda), tf.float32)).numpy()
            if divergencia_ratio > 0.05:
                log(f"[CONSENSO] Suprema diverge da maioria em {divergencia_ratio*100:.2f}% dos pixels — veto epistêmico aplicado.")
                return 0.0

    votos_stacked_tensor = tf.stack(votos_stacked)  # (N, 30, 30, 10)
    pesos_tensor = tf.constant(pesos_usados, dtype=tf.float32)  # (N,)

    consenso_total = 0.0
    total_pixels = 30 * 30 * 10

    for i in range(30):
        for j in range(30):
            for k in range(10):
                valores_pixel = votos_stacked_tensor[:, i, j, k]  # (N,)
                pesos_pixel = tf.zeros_like(pesos_tensor)
                valor_referencia = valores_pixel[0]
                for idx, valor in enumerate(valores_pixel):
                    if valor == valor_referencia:
                        pesos_pixel = tf.tensor_scatter_nd_update(pesos_pixel, [[idx]], [pesos_tensor[idx]])
                soma_pesos = tf.reduce_sum(pesos_pixel).numpy()
                if soma_pesos >= required_score:
                    consenso_total += 1

    consenso_final = consenso_total / total_pixels
    log(f"[CONSENSO PONDERADO] {consenso_final*100:.2f}% dos pixels atingiram pontuação mínima de {required_score}")
    return consenso_final
