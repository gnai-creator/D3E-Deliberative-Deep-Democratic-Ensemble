import tensorflow as tf
import numpy as np
from runtime_utils import log

class ConfidenceManager:
    def __init__(self, models: dict, initial_confidence=0.5):
        self.model_names = list(models.keys())
        self.confidences = {name: initial_confidence for name in self.model_names}

    def get_confidence(self, name):
        return self.confidences.get(name, 0.0)

    def get_active_model_names(self, threshold=0.5):
        return [name for name, conf in self.confidences.items() if conf >= threshold]

    def update_confidences(self, votos_models: dict, voto_referencia, acerto_bonus=0.05, erro_penalidade=0.1):
        """
        votos_models: dict com nome do modelo -> logits
        voto_referencia: tensor de shape (1, H, W), valores inteiros (classe correta esperada)
        """
        for name, logits in votos_models.items():
            pred = tf.argmax(logits, axis=-1)  # (1, H, W)
            match = tf.equal(pred, voto_referencia)
            acertos = tf.reduce_sum(tf.cast(match, tf.float32)).numpy()
            total = tf.size(pred).numpy()
            acc = acertos / total

            if acc > 0.9:
                self.confidences[name] = min(1.0, self.confidences[name] + acerto_bonus)
            elif acc < 0.7:
                self.confidences[name] = max(0.0, self.confidences[name] - erro_penalidade)

    def reabilitar_modelos(self, boost=0.05, max_conf=0.3):
        """
        Dá uma pequena recuperação para modelos muito desacreditados (com confiança abaixo de max_conf)
        """
        for name, conf in self.confidences.items():
            if conf < max_conf:
                self.confidences[name] += boost
                self.confidences[name] = min(1.0, self.confidences[name])

    def print_status(self):
        log("[CONFIDENCE STATUS]")
        for name, conf in self.confidences.items():
            log(f" - {name}: {conf:.2f}")


def avaliar_consenso_com_confiança(votos_models: dict, confidence_manager, required_votes=5, confidence_threshold=0.5):
    """
    votos_models: dict com nomes dos modelos como chaves e tensores de logits como valores
    confidence_manager: instância de ConfidenceManager
    required_votes: número mínimo de votos similares para haver consenso por pixel
    confidence_threshold: nível mínimo de confiança para um voto ser considerado

    Retorna: consenso médio por pixel (float entre 0 e 1)
    """
    # Filtra os modelos confiáveis
    ativos = confidence_manager.get_active_model_names(threshold=confidence_threshold)
    logits_ativos = [v for k, v in votos_models.items() if k in ativos]

    if not logits_ativos:
        log("[AVISO] Nenhum modelo com confiança suficiente para avaliação de consenso.")
        return 0.0

    votos_classe = [tf.argmax(v, axis=-1) for v in logits_ativos]  # cada um: (1, H, W)

    # Remove dimensão 1 (batch) se presente
    votos_classe = [tf.squeeze(v, axis=0) if v.shape.rank == 3 else v for v in votos_classe]

    votos_stacked = tf.stack(votos_classe, axis=0)  # (N, H, W)
    if votos_stacked.shape.rank != 3:
        raise ValueError(f"[ERRO] Shape inesperado em votos_stacked: {votos_stacked.shape}. Esperado (N, H, W)")

    def contar_consenso(votos_pixel):
        uniques, _, count = tf.unique_with_counts(votos_pixel)
        return tf.reduce_max(count)

    votos_majoritarios = tf.map_fn(
        lambda x: tf.map_fn(lambda y: contar_consenso(y), x, dtype=tf.int32),
        tf.transpose(votos_stacked, [1, 2, 0]),  # (H, W, N)
        dtype=tf.int32
    )

    consenso_bin = tf.cast(votos_majoritarios >= required_votes, tf.float32)
    consenso_final = tf.reduce_mean(consenso_bin).numpy()

    log(f"[CONSENSO - COM CONFIANÇA] {len(logits_ativos)} modelos contribuíram para o consenso:")
    for name in ativos:
        log(f" - {name}: {confidence_manager.get_confidence(name):.2f}")

    return consenso_final
