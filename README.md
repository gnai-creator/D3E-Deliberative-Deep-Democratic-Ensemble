# SimuV1 ARC Court AI System

Este projeto é uma simulação avançada de julgamento automático de tarefas do [ARC Challenge](https://github.com/fchollet/ARC). Ele introduz uma arquitetura inspirada em um tribunal, composta por múltiplos modelos (juradas, advogada, juíza e suprema juíza), onde a tomada de decisão é baseada em consenso.

## Estrutura do Tribunal

- **Juradas (3 modelos):** Aprendem com a saída da Advogada.
- **Advogada:** Aprende com a saída da Suprema Juíza.
- **Juíza:** Aprende com as saídas das Juradas e da Advogada.
- **Suprema Juíza:** Treina com as predições da Juíza, decide o veredito final e re-treina a Advogada.

## Arquiteturas SimuV*

O projeto contém cinco versões de modelos neurais sofisticados:

- `SimuV1.py` a `SimuV5.py`: Modelos sequenciais com módulos atencionais, permutadores de classe, codificações posicionais e módulos de memória.
- Usam a API do TensorFlow 2.x e Keras.

## Requisitos

- Python 3.10
- TensorFlow >= 2.12
- Pandas, NumPy, Matplotlib, Seaborn
- OpenCV (para geração de vídeos)

## Execução

### 1. Pré-processamento dos Dados

```bash
python3 main.py --prepare
```

### 2. Treinamento e Julgamento

```bash
python3 main.py --train
```

### 3. Teste no ARC Challenge

```bash
python3 main.py --test_challenge
```

## Estrutura dos Diretórios

- `main.py`: Arquivo principal de execução.
- `court_logic.py`: Lógica da simulação do tribunal com treinamento iterativo.
- `metrics_utils.py`: Visualização dos votos e métricas de consenso.
- `data_preparation.py`: Carregamento e formatação dos dados do ARC.
- `SimuV*.py`: Arquivos com diferentes versões dos modelos neurais.
- `neural_blocks.py`: Blocos personalizados usados por todos os modelos.
- `votos_visuais/`: Gera visualizações dos votos por iteração.
- `videos/`: Armazena vídeos compilados com as votações visuais.

## Modo de Consenso

O sistema considera consenso quando **5 dos 6 modelos** concordam com a mesma classe para um pixel. A Suprema Juíza treina até que sua loss seja inferior a `0.05` e a acurácia seja superior a `0.95`.

## Visualizações

Cada iteração gera um arquivo `.png` com a predição de cada modelo e um mapa de consenso, salvos em `votos_visuais/`. Esses arquivos são usados para criar vídeos da deliberação.

## Erros Comuns

- `ValueError: Entrada com shape inesperado`: certifique-se que todos os tensores estão com o shape [B, 30, 30, 1, C].
- `UnboundLocalError: votos_models_final`: ocorre quando não há votos válidos durante iterações iniciais. Proteja o código com verificações.

## Autor

Desenvolvido por [gnai-creator](https://github.com/gnai-creator), com amor e paciência para lidar com modelos que argumentam mais que humanos.

---

**Nota:** Este projeto é uma simulação experimental com fins de pesquisa. Não use isso para tomar decisões jurídicas no mundo real. (Ainda.)
