# deliberacao_hierarquica.py
from jurado_linguagem import JuradoLinguagem
from analise_argumentos import gerar_embedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from runtime_utils import log

# Pesos por papel no tribunal
PESOS = {
    "Jurado": 1.0,
    "Advogado": 2.0,
    "Juiz": 3.0,
    "Promotor": 3.0,
    "Supremo": 5.0
}

class MembroTribunal:
    def __init__(self, papel):
        self.nome = papel
        self.modelo = JuradoLinguagem(nome=papel)
        self.fala = None
        self.embedding = None

    def deliberar(self, grid, argumento_previo=None):
        self.fala = self.modelo.deliberar(grid, argumento_previo)
        self.embedding = gerar_embedding(self.fala)
        return self.fala

def calcular_media_ponderada(vetores, pesos):
    total_peso = 0.0
    soma_ponderada = None
    for nome, vetor in vetores.items():
        peso = pesos.get(nome, 1.0)
        if soma_ponderada is None:
            soma_ponderada = peso * vetor
        else:
            soma_ponderada += peso * vetor
        total_peso += peso
    return soma_ponderada / total_peso

def avaliar_alinhamento(falas, vetor_consenso):
    log("[ALINHAMENTO SEMÂNTICO]")
    for nome, vetor in falas.items():
        if vetor_consenso is None or vetor is None:
            log(f"[{nome}] ❌ Sem vetor de embedding para avaliar.")
            continue

        vetor_np = vetor.detach().cpu().numpy() if hasattr(vetor, "detach") else np.asarray(vetor)
        consenso_np = vetor_consenso.detach().cpu().numpy() if hasattr(vetor_consenso, "detach") else np.asarray(vetor_consenso)

        similaridade = cosine_similarity([vetor_np], [consenso_np])[0][0]
        status = "✅ Alinhado" if similaridade > 0.85 else "⚠️ Parcial" if similaridade > 0.65 else "❌ Em desacordo"
        log(f"[{nome}] Similaridade com o consenso: {similaridade:.2f} → {status}")

def executar_rodada_deliberacao(grid):
    membros = [
        MembroTribunal("Jurado"),
        MembroTribunal("Advogado"),
        MembroTribunal("Juiz"),
        MembroTribunal("Promotor"),
        MembroTribunal("Supremo"),
    ]

    falas = {}
    argumento = None
    membro_anterior = None

    for membro in membros:
        if argumento:
            argumento_formatado = f"O {membro_anterior} disse: {argumento}. O que você acha?"
        else:
            argumento_formatado = None

        fala = membro.deliberar(grid, argumento_previo=argumento_formatado)
        log(f"[{membro.nome}] disse: \"{fala}\"")

        argumento = fala
        membro_anterior = membro.nome
        falas[membro.nome] = membro.embedding

    vetor_consenso = calcular_media_ponderada(falas, PESOS)
    avaliar_alinhamento(falas, vetor_consenso)

# Exemplo de uso (no main.py)
# from deliberacao_hierarquica import executar_rodada_deliberacao
# executar_rodada_deliberacao(grid_de_exemplo)
