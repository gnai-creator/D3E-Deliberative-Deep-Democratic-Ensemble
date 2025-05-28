import os
import tensorflow as tf
import numpy as np
import random
from GrampyX import rodar_deliberacao_com_condicoes, GrampyX
from models_loader import load_model
from runtime_utils import log
if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    idx = 0
    grampyx_instance = GrampyX()

    while True:
        sucesso = rodar_deliberacao_com_condicoes(
            parar_se_sucesso=True,
            max_iteracoes=150,
            consenso_minimo=0.9,
            idx=idx,
            grampyx=grampyx_instance
        )
        log(f"[RELOAD] INICIANDO NOVA RODADA IDX={idx}")
        if sucesso:
            log(f"[RELOAD] consenso em idx={idx}. Recarregando modelos.")
            grampyx_instance.models = [load_model(i, 0.0005) for i in range(grampyx_instance.num_modelos)]

        idx += 1
