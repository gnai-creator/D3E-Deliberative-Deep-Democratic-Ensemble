# analise_argumentos.py
from sentence_transformers import SentenceTransformer, util
import torch
from runtime_utils import log

# Carrega um modelo eficiente e especializado em similaridade semântica
log("[INFO] Carregando modelo de embeddings semânticos...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
log("[INFO] Modelo carregado com sucesso!")


def gerar_embedding(frase):
    """Gera o embedding vetorial da frase."""
    with torch.no_grad():
        embedding = model.encode(frase, convert_to_tensor=True)
    return embedding


def calcular_similaridade(frase1: str, frase2: str) -> float:
    """Calcula a similaridade entre duas frases usando cosseno."""
    emb1 = gerar_embedding(frase1)
    emb2 = gerar_embedding(frase2)
    sim = util.cos_sim(emb1, emb2).item()
    return sim
