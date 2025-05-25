

📌 Próximos passos (Fine Tuning):
Avaliar com mais dados de teste para garantir generalização.

Ajustar thresholds de confiança (tipo confidence_threshold=0.5) para ver como isso impacta o consenso.

Dar “redireito ao voto” para modelos com confiança que sobe — já tem estrutura pra isso.

Melhorar a robustez do input para casos ambíguos.

Aplicar regularização no treinamento da Juíza Suprema se quiser evitar overfitting em poucas amostras.

Talvez revisar o loss function de alguns dos modelos simples (se quiser que todos votem mais parecido).

Benchmarkar tempo de julgamento caso use isso em larga escala.

