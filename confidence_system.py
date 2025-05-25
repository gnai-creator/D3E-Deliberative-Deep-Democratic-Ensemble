import numpy as np
import tensorflow as tf
from runtime_utils import log

# Inicializa a confiança dos 6 modelos
def init_confidence(n=6):
    # Começa com confiança 1 para todos
    return np.ones(n)

# Atualiza a confiança com base em concordância com a Suprema Juíza
def update_confidence(conf, votos_models, votos_supremos, decaimento=0.1):
    new_conf = conf.copy()
    votos_classes = [tf.argmax(v, axis=-1).numpy() for v in votos_models]
    votos_supremos_np = votos_supremos.numpy() if isinstance(votos_supremos, tf.Tensor) else votos_supremos

    for i in range(len(votos_models)):
        if conf[i] == 0:
            continue  # modelo sem direito ao voto
        voto = votos_classes[i]
        acordo = (voto == votos_supremos_np)
        taxa_acordo = np.sum(acordo) / acordo.size
        if taxa_acordo < 0.6:
            new_conf[i] = max(0, new_conf[i] - decaimento)
            log(f"[CONF] Modelo {i} perdeu confiança. Taxa de acordo: {taxa_acordo:.2f}")
        else:
            log(f"[CONF] Modelo {i} manteve confiança. Taxa de acordo: {taxa_acordo:.2f}")
    return new_conf

# Recupera confiança de modelos que estavam fora
def restore_confidence(conf, votos_models, votos_supremos, aumento=0.05):
    new_conf = conf.copy()
    votos_classes = [tf.argmax(v, axis=-1).numpy() for v in votos_models]
    votos_supremos_np = votos_supremos.numpy() if isinstance(votos_supremos, tf.Tensor) else votos_supremos

    for i in range(len(votos_models)):
        if conf[i] >= 1:
            continue
        voto = votos_classes[i]
        acordo = (voto == votos_supremos_np)
        taxa_acordo = np.sum(acordo) / acordo.size
        if taxa_acordo > 0.7:
            new_conf[i] = min(1, new_conf[i] + aumento)
            log(f"[RECUPERAÇÃO] Modelo {i} recuperou parte da confiança. Taxa de acordo: {taxa_acordo:.2f}")
    return new_conf

# (opcional) Verifica se há votos válidos
def get_valid_voters(conf):
    return [i for i, c in enumerate(conf) if c > 0]

# (opcional) Visualiza o estado de confiança em string
def print_confidence_bar(conf):
    return " | ".join([f"M{i}:{c:.2f}" for i, c in enumerate(conf)])
