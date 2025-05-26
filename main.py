import os
import tensorflow as tf
import numpy as np
import random
from GrampyX import rodar_deliberacao_com_condicoes
if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    idx = 0
    while True:
        while rodar_deliberacao_com_condicoes(
            parar_se_sucesso=True,
            max_iteracoes=150,
            consenso_minimo=0.9,
            idx=idx
        ):
            idx += 1