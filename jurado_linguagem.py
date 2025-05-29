# jurado_linguagem.py
from model_loader import tokenizer, model
import torch
import random
import numpy as np

class JuradoLinguagem:
    def __init__(self, nome="jurado"):
        self.nome = nome
        self.historico = []
        self.device = model.device if hasattr(model, "device") else next(model.parameters()).device

    def descrever_padrao(self, grid):
        if grid == [row[::-1] for row in grid]:
            return "O padrão apresenta simetria horizontal."
        if grid == grid[::-1]:
            return "O padrão apresenta simetria vertical."
        if all(all(cell == row[0] for cell in row) for row in grid):
            return "Todas as linhas possuem o mesmo valor."
        if all(all(grid[i][j] == grid[0][j] for i in range(len(grid))) for j in range(len(grid[0]))):
            return "Todas as colunas possuem o mesmo valor."
        if all(grid[i][i] == grid[0][0] for i in range(min(len(grid), len(grid[0])))):
            return "O padrão apresenta simetria diagonal principal."
        if all(grid[i][len(grid[0])-1-i] == grid[0][len(grid[0])-1] for i in range(min(len(grid), len(grid[0])))):
            return "O padrão apresenta simetria diagonal secundária."
        return "Não identifiquei uma simetria clara."

    def gerar_hipotese(self, descricao):
        if "simetria horizontal" in descricao:
            return "Proponho espelhar a última linha horizontalmente."
        if "simetria vertical" in descricao:
            return "Proponho repetir a primeira coluna invertida na última."
        if "linhas possuem o mesmo valor" in descricao:
            return "Sugiro repetir a linha padrão nas posições faltantes."
        if "colunas possuem o mesmo valor" in descricao:
            return "Sugiro repetir a coluna padrão nas posições faltantes."
        if "simetria diagonal principal" in descricao:
            return "Sugiro replicar o valor da diagonal principal em cruz."
        if "simetria diagonal secundária" in descricao:
            return "Sugiro replicar a diagonal secundária para completar o padrão."
        return "Sugiro repetir o padrão mais próximo ao anterior."

    def responder_argumento(self, argumento_outro):
        respostas = [
            "Discordo, pois a posição central não bate com sua proposta.",
            "Concordo parcialmente, mas há uma variação no padrão inferior.",
            "Seu argumento ignora a coloração do lado direito.",
        ]
        return random.choice(respostas)

    def gerar_texto_com_llm(self, prompt, grid, argumento_previo):
        mensagem = [
            {"role": "system", "content": "Você é um membro do tribunal avaliando padrões em uma matriz."},
            {"role": "user", "content": argumento_previo if argumento_previo else "Observe o padrão abaixo e faça uma análise."},
            {"role": "user", "content": f"Grid: {grid}\nDiga sua observação e conclusão em até 3 parágrafos."}
        ]

        inputs = tokenizer.apply_chat_template(mensagem, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.no_grad():
            output_ids = model.generate(
                inputs,
                max_new_tokens=250,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=1.2,
                top_p=0.95
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def gerar_embedding(self, frase):
        inputs = tokenizer(frase, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Última camada
            embedding = hidden_states.mean(dim=1)      # Média dos tokens
        return embedding.squeeze().cpu().numpy()

    def sanitizar_fala(self, fala):
        linhas = fala.splitlines()
        únicas = []
        vistos = set()
        for linha in linhas:
            if linha not in vistos:
                únicas.append(linha)
                vistos.add(linha)
        return "\n".join(únicas)

    def deliberar(self, grid, argumento_previo=None):
        descricao = self.descrever_padrao(grid)
        hipotese = self.gerar_hipotese(descricao)
        resposta = ""
        if argumento_previo:
            resposta = self.responder_argumento(argumento_previo)

        prompt = f"{self.nome} observa o padrão: '{descricao}'. {hipotese} {resposta}\nResposta final:"
        fala = self.gerar_texto_com_llm(prompt, grid, argumento_previo)
        fala = self.sanitizar_fala(fala)

        embedding = self.gerar_embedding(fala)
        self.historico.append((fala, embedding))
        return fala
