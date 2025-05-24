arc-court-ai/
│
├── models/
│   ├── ia1.py       # Levemente enviesada pra simetria
│   ├── ia2.py       # Mais foco em cor
│   ├── ia3.py       # Ignora posições, mais topológica
│   ├── ia4.py       # Estável, referência de treino
│   └── judge.py     # A IA juíza — recebe outputs, julga juízo
│
├── court_logic.py   # Coordena o ciclo de decisão
├── train_all.py     # Treina todas as IAs com o dataset base
├── inference.py     # Roda os testes com votação em pipeline
│
├── data/
│   └── arc-agi_test_challenges.json
│
└── README.md        # Instruções pra malucos como você
