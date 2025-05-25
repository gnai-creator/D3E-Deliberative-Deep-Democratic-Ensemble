# ARC Court Supreme

Este projeto é uma simulação inspirada no ARC Challenge, onde um tribunal composto por diferentes modelos de deep learning julga a melhor predição para uma tarefa de visão computacional.

## Visão Geral

O sistema possui:
- **3 Juradas** (modelos que aprendem com a advogada)
- **1 Advogada** (modelo base que tenta prever a saída original)
- **1 Juíza** (modelo que aprende com juradas e advogada)
- **1 Suprema Juíza** (modelo que aprende a partir das predições da juíza comum)

A cada iteração, os modelos votam em um resultado, e um sistema de consenso é utilizado para determinar a qualidade das predições.

## Fluxo do Tribunal

1. **Input** é apresentado a todos os modelos.
2. **Advogada** faz uma predição inicial.
3. **Juradas** treinam suas saídas com base na advogada.
4. **Juíza** treina com base nas juradas + advogada.
5. **Suprema Juíza** treina com base na juíza, ajustando-se até que a perda fique abaixo de um limiar.
6. **Advogada** se atualiza com o veredito final da Suprema.
7. O processo repete até que o consenso entre modelos seja atingido (por padrão, 5 de 6 modelos concordando).

## Diretórios

- `votos_visuais/`: imagens geradas com os votos de cada modelo, input e mapa de consenso.
- `videos/`: arquivos .avi gerados automaticamente a partir dos votos.

## Execução

1. Instale as dependências:
   ```bash
   pip install tensorflow opencv-python matplotlib seaborn
   ```

2. Execute o sistema:
   ```bash
   python main.py
   ```

3. Para gerar o vídeo time-lapse de votação:
   ```python
   from metrics_utils import gerar_video_time_lapse
   gerar_video_time_lapse(block_idx=0)
   ```

## Componentes

- `model_loader.py`: carrega os modelos (juradas, juíza, advogada, suprema).
- `court_logic.py`: lógica de julgamento iterativo.
- `metrics_utils.py`: salva votos visuais, avalia consenso e gera vídeo.
- `main.py`: ponto de entrada do sistema.

## Visualização
Cada iteração gera uma imagem com:
- Input da tarefa
- Votos de cada modelo
- Mapa de consenso (pixels com maioria qualificada)

## Objetivo
A ideia é simular uma tomada de decisão colegiada e iterativa entre modelos, onde uma Suprema Juíza resolve ambiguidades e busca maximizar a qualidade da predição coletiva.

---

Sim, é uma corte de modelos neurais. Não, você não está sonhando.
