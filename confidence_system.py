import tensorflow as tf

class ConfidenceManager:
    def __init__(self, model_dict, initial_confidence=0.5, min_confidence=0.0, max_confidence=1.0, adjustment=0.05):
        """
        model_dict: dicionário {nome: modelo}
        """
        self.model_names = list(model_dict.keys())
        self.confidences = {name: initial_confidence for name in self.model_names}
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.adjustment = adjustment

    def get_confidence(self, model_name):
        return self.confidences.get(model_name, 0.0)

    def get_active_model_names(self, threshold=0.5):
        return [name for name, conf in self.confidences.items() if conf >= threshold]

    def update_confidences(self, votos_models, votos_supremos):
        import tensorflow as tf

        votos_supremos = tf.cast(votos_supremos, tf.int64)

        for name, logits in votos_models.items():
            pred_classes = tf.argmax(logits, axis=-1)
            match = tf.cast(pred_classes == votos_supremos, tf.float32)
            acc = tf.reduce_mean(match).numpy()

            # Ajusta a confiança com base na acurácia
            if acc > 0.5:
                self.confidences[name] = min(self.max_confidence, self.confidences[name] + self.adjustment)
            else:
                self.confidences[name] = max(self.min_confidence, self.confidences[name] - self.adjustment)

        # Log opcional
        print("[CONFIDENCE UPDATE]")
        for name in self.model_names:
            print(f" - {name}: {self.confidences[name]:.2f}")

def safe_squeeze_last_axis(tensor):
    return tf.squeeze(tensor, axis=-1) if tensor.shape.rank == 3 and tensor.shape[-1] == 1 else tensor


def avaliar_consenso_com_confiança(votos_models: dict, confidence_manager, required_votes=5, confidence_threshold=0.5):
    """
    votos_models: dict com nomes dos modelos como chaves e tensores de logits como valores
    confidence_manager: instância de ConfidenceManager
    required_votes: número mínimo de votos similares para haver consenso por pixel
    confidence_threshold: nível mínimo de confiança para um voto ser considerado

    Retorna: consenso médio por pixel (float entre 0 e 1)
    """
    import tensorflow as tf

    # Obtém modelos ativos e confiáveis
    active_names = confidence_manager.get_active_model_names()
    logits_ativos = {
        name: logits for name, logits in votos_models.items()
        if name in active_names and confidence_manager.get_confidence(name) >= confidence_threshold
    }

    if not logits_ativos:
        print("[AVISO] Nenhum modelo com confiança suficiente para avaliação de consenso.")
        return 0.0

    # Garantir que os logits tenham shape [1, H, W, C]
    votos_classe = []
    for name, logits in logits_ativos.items():
        pred = tf.argmax(logits, axis=-1)
        if pred.shape.rank == 4:
            pred = tf.squeeze(pred, axis=0)
        elif pred.shape.rank == 3:
            pass  # já está em [H, W, C]
        elif pred.shape.rank == 2:
            pred = tf.expand_dims(pred, axis=-1)
        else:
            raise ValueError(f"[ERRO] Shape inesperado para predição de {name}: {pred.shape}")
        votos_classe.append(pred)

    # Garantir que todas as previsões têm shape [H, W]
    votos_classe = [
        tf.squeeze(tf.argmax(logits, axis=-1), axis=0)
        if tf.rank(tf.argmax(logits, axis=-1)) == 3
        else tf.argmax(logits, axis=-1)
        for logits in logits_ativos.values()
    ]


    votos_stacked = tf.stack(votos_classe, axis=0)  # Shape: [num_modelos, H, W]

    if votos_stacked.shape.rank != 3:
        raise ValueError(f"[ERRO] Shape inesperado em votos_stacked: {votos_stacked.shape}. Esperado (N, H, W)")

    # Define função para contar o número de votos majoritários por pixel
    def contar_consenso(votos_pixel):
        uniques, _, count = tf.unique_with_counts(votos_pixel)
        return tf.reduce_max(count)

    # Aplica função em cada pixel
    votos_majoritarios = tf.map_fn(
        lambda x: tf.map_fn(lambda y: contar_consenso(y), x, dtype=tf.int32),
        tf.transpose(votos_stacked, [1, 2, 0]),  # [H, W, num_modelos]
        dtype=tf.int32
    )

    consenso_bin = tf.cast(votos_majoritarios >= required_votes, tf.float32)
    consenso_final = tf.reduce_mean(consenso_bin).numpy()

    print(f"[CONSENSO - COM CONFIANÇA] {len(logits_ativos)} modelos contribuíram para o consenso")
    for name in logits_ativos:
        conf = confidence_manager.get_confidence(name)
        print(f" - {name} (confiança: {conf:.2f})")

    return consenso_final
