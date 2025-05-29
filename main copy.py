import os
import numpy as np
import random
from TribunalSimbolico import TribunalSimbolico
import torch
from deliberacao_hierarquica import executar_rodada_deliberacao
# executar_rodada_deliberacao(grid_de_exemplo)
if __name__ == "__main__":


    exemplo_grid = [
        [1, 2, 1],
        [1, 2, 1],
        [1, 2, 1],
    ]
    executar_rodada_deliberacao(exemplo_grid)  