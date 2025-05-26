import tensorflow as tf

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
            print(linha)

def avaliar_consenso_com_confiança(votos_models: dict, confidence_manager, required_votes=5, confidence_threshold=0.5):
    """
    votos_models: dict com nomes dos modelos como chaves e tensores de logits como valores
    confidence_manager: instância de ConfidenceManager
    required_votes: número mínimo de votos similares para haver consenso por pixel
    confidence_threshold: nível mínimo de confiança para um voto ser considerado

    Retorna: consenso médio por pixel (float entre 0 e 1)
    """
    active_names = confidence_manager.get_active_model_names(threshold=confidence_threshold)

    if not active_names:
        print("[CONSENSO] Nenhum modelo com confiança suficiente.")
        return 0.0

    votos_classe = []
    for name in active_names:
        logits = votos_models[name]
        voto = tf.argmax(logits, axis=-1)
        if voto.shape.rank > 3:
            voto = tf.squeeze(voto, axis=0)
        votos_classe.append(voto)

    votos_stacked = []
    for name, v in votos_models.items():
        try:
            v = tf.argmax(v, axis=-1)
            if v.shape.rank > 3:
                v = tf.squeeze(v, axis=0)
            if tf.size(v) != 900:
                print(f"[CONSENSO] ⚠️ Voto {name} tem {tf.size(v).numpy()} elementos. Ignorado.")
                continue
            v = tf.reshape(v, (30, 30))
            votos_stacked.append(v)
        except Exception as e:
            print(f"[CONSENSO] Erro ao processar voto de {name}: {e}")

    if not votos_stacked:
        print("[CONSENSO] Nenhum voto válido após filtragem.")
        return 0.0

    votos_stacked = tf.stack(votos_stacked)


    if votos_stacked.shape.rank != 3:
        raise ValueError(f"[ERRO] Shape inesperado em votos_stacked: {votos_stacked.shape}. Esperado (N, H, W)")

    def contar_consenso(votos_pixel):
        uniques, _, count = tf.unique_with_counts(votos_pixel)
        return tf.reduce_max(count)

    votos_majoritarios = tf.map_fn(
        lambda x: tf.map_fn(lambda y: contar_consenso(y), x, dtype=tf.int32),
        tf.transpose(votos_stacked, [1, 2, 0]),  # [H, W, N]
        dtype=tf.int32
    )

    consenso_bin = tf.cast(votos_majoritarios >= required_votes, tf.float32)
    consenso_final = tf.reduce_mean(consenso_bin).numpy()

    print(f"[CONSENSO - COM CONFIANÇA] {len(votos_classe)} modelos válidos contribuíram (≥{confidence_threshold:.2f})")
    for name in active_names:
        conf = confidence_manager.get_confidence(name)
        print(f" - {name}: {conf:.3f}")

    return consenso_final
