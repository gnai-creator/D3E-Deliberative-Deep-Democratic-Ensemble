# test_argumentacao_semantica.py
from jurado_linguagem import JuradoLinguagem
from analise_argumentos import calcular_similaridade
from runtime_utils import log
class TribunalSimbolico:
    def __init__(self, grid):
        self.grid = grid
        self.jurados = [
            JuradoLinguagem("Jurado_0"),
            JuradoLinguagem("Jurado_1")
        ]
        self.falas = []

    def executar_deliberacao(self):
        fala_0 = self.jurados[0].deliberar(self.grid)
        self.falas.append(fala_0)
        log(fala_0)

        fala_1 = self.jurados[1].deliberar(self.grid, argumento_previo=fala_0)
        self.falas.append(fala_1)
        log(fala_1)

        self.analisar_argumentos()

    def analisar_argumentos(self):
        if len(self.falas) < 2:
            log("[ERRO] Número insuficiente de falas para análise.")
            return

        log("\n[INFO] Analisando similaridade semântica com modelo 'all-MiniLM-L6-v2'...")
        sim = calcular_similaridade(self.falas[0], self.falas[1])
        log(f"[ANÁLISE SEMÂNTICA] Similaridade entre os jurados: {sim:.2f}")

        if sim > 0.85:
            log("\n✅ Jurados estão semanticamente alinhados.")
        elif sim > 0.65:
            log("\n⚠️ Jurados têm alguma convergência semântica.")
        else:
            log("\n❌ Jurados discordam semântica e argumentativamente.")



