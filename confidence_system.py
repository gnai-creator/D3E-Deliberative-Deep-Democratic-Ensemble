import tensorflow as tf
from runtime_utils import log
from metrics_utils import safe_squeeze

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

def flatten_voto_simbólico(v):
    v = tf.convert_to_tensor(v)
    if v.shape.rank > 2 and v.shape[-1] > 1:
        v = tf.argmax(v, axis=-1)
    if v.shape.rank > 2:
        v = safe_squeeze(v, axis=0)
    return tf.reshape(v, (1, -1))

def avaliar_consenso_com_confianca(votos_models: dict, confidence_manager, required_votes=5, confidence_threshold=0.5, voto_reverso_ok=None):
    active_names = confidence_manager.get_active_model_names(threshold=confidence_threshold)

    if not active_names:
        log("[CONSENSO] Nenhum modelo com confiança suficiente.")
        return 0.0

    votos_stacked = []
    votos_dict = {}
    for name in active_names:
        try:
            v = votos_models[name]
            v = tf.convert_to_tensor(v)
            if v.shape[-1] > 1:
                v = tf.argmax(v, axis=-1)
            v = tf.squeeze(v)
            if voto_reverso_ok and name in voto_reverso_ok:
                v = 9 - v
            if tf.size(v) != 900:
                log(f"[CONSENSO] ⚠️ Voto {name} tem {tf.size(v).numpy()} elementos. Ignorado.")
                continue
            v = tf.reshape(v, (30, 30))
            votos_stacked.append(v)
            votos_dict[name] = v
        except Exception as e:
            log(f"[CONSENSO] Erro ao processar voto de {name}: {e}")

    if not votos_stacked:
        log("[CONSENSO] Nenhum voto válido após filtragem.")
        return 0.0

    votos_stacked_tensor = tf.stack(votos_stacked)

    if votos_stacked_tensor.shape.rank != 3:
        raise ValueError(f"[ERRO] Shape inesperado em votos_stacked: {votos_stacked_tensor.shape}. Esperado (N, H, W)")

    votos_moda = tf.cast(tf.round(tf.reduce_mean(votos_stacked_tensor, axis=0)), tf.int32)

    for name in votos_dict:
        acertou = tf.reduce_all(tf.equal(tf.cast(votos_dict[name], tf.int64), tf.cast(votos_moda, tf.int64))).numpy()
        confidence_manager.update_confidence(name, acertou)

    def contar_consenso(votos_pixel):
        uniques, _, count = tf.unique_with_counts(votos_pixel)
        return tf.reduce_max(count)

    votos_majoritarios = tf.map_fn(
        lambda x: tf.map_fn(lambda y: contar_consenso(y), x, dtype=tf.int32),
        tf.transpose(votos_stacked_tensor, [1, 2, 0]),
        dtype=tf.int32
    )

    consenso_bin = votos_majoritarios >= required_votes
    consenso_final = float(tf.reduce_all(consenso_bin).numpy())



    log(f"[CONSENSO - COM CONFIANÇA] {len(votos_stacked)} modelos válidos contribuíram (≥{confidence_threshold:.2f})")
    for name in active_names:
        conf = confidence_manager.get_confidence(name)
        log(f" - {name}: {conf:.3f}")

    return consenso_final

# consensus_system_weighted.py — Consenso ponderado por hierarquia de modelos

def avaliar_consenso_ponderado(votos_models: dict, pesos: dict, required_score=5.0, voto_reverso_ok=None):
    """
    Avalia consenso ponderado usando pesos hierárquicos definidos para cada modelo.
    Cada pixel é decidido com base na soma dos pesos dos modelos que concordam naquele ponto.
    """
    votos_stacked = []
    votos_dict = {}
    pesos_usados = []

    for name, voto in votos_models.items():
        try:
            v = tf.convert_to_tensor(voto)
            if v.shape[-1] > 1:
                v = tf.argmax(v, axis=-1)
            v = tf.cast(tf.squeeze(v), tf.int64)
            if voto_reverso_ok and name in voto_reverso_ok:
                v = 9 - v
            if tf.size(v) != 900:
                log(f"[CONSENSO] ⚠️ Voto {name} tem {tf.size(v).numpy()} elementos. Ignorado.")
                continue
            v = tf.reshape(v, (30, 30))
            votos_stacked.append(v)
            votos_dict[name] = v
            pesos_usados.append(pesos.get(name, 1.0))
        except Exception as e:
            log(f"[CONSENSO] Erro ao processar voto de {name}: {e}")

    if not votos_stacked:
        log("[CONSENSO] Nenhum voto válido após filtragem.")
        return 0.0

    # Suprema tem poder de veto simbólico se divergir da maioria
    if "modelo_5" in votos_dict:
        voto_suprema = tf.cast(votos_dict["modelo_5"], tf.int64)
        votos_sem_suprema = [tf.cast(v, tf.int64) for k, v in votos_dict.items() if k != "modelo_5" and k != "modelo_6"]
        if votos_sem_suprema:
            votos_moda = tf.cast(tf.round(tf.reduce_mean(tf.stack(votos_sem_suprema), axis=0)), tf.int64)
            divergencia = tf.reduce_any(tf.not_equal(voto_suprema, votos_moda))
            if divergencia:
                log("[CONSENSO] Suprema diverge da maioria — veto epistêmico aplicado.")
                return 0.0

    votos_stacked_tensor = tf.stack(votos_stacked)  # (N, 30, 30)
    pesos_tensor = tf.constant(pesos_usados, dtype=tf.float32)  # (N,)

    consenso_total = 0.0
    total_pixels = 30 * 30

    for i in range(30):
        for j in range(30):
            valores_pixel = votos_stacked_tensor[:, i, j]  # (N,)
            pesos_pixel = tf.zeros_like(pesos_tensor)
            for idx, valor in enumerate(valores_pixel):
                if tf.reduce_all(valor == valores_pixel):
                    pesos_pixel = tf.tensor_scatter_nd_update(pesos_pixel, [[idx]], [pesos_tensor[idx]])
            soma_pesos = tf.reduce_sum(pesos_pixel).numpy()
            if soma_pesos >= required_score:
                consenso_total += 1

    consenso_final = consenso_total / total_pixels
    log(f"[CONSENSO PONDERADO] {consenso_final*100:.2f}% dos pixels atingiram pontuação mínima de {required_score}")
    return consenso_final