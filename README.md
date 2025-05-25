![D3E - Deliberative Deep Democratic Ensemble](DDDE.png)

# D3E: Deliberative Deep Democratic Ensemble

D3E (Deliberative Deep Democratic Ensemble) é um sistema de aprendizado coletivo inspirado na estrutura de um tribunal. Foi projetado para resolver tarefas do ARC Challenge com base em deliberacão iterativa entre modelos neurais especializados.

## 🏛️ Arquitetura

* **Juradas (3)**: Modelos que aprendem com a advogada.
* **Advogada (1)**: Aprende com os dados crus (raw input) e a Suprema Juíza.
* **Juíza (1)**: Agrega a opinião das juradas e da advogada.
* **Suprema Juíza (1)**: Refina a predição com base apenas no input original, em ciclos iterativos, até que haja consenso entre pelo menos 5 modelos.

## ⚖️ Ciclo de Julgamento

1. O input (do ARC) é apresentado.
2. A advogada faz uma predição com base apenas no input.
3. Juradas são treinadas com base na predição da advogada.
4. A juíza se treina com as saídas das juradas e da advogada.
5. Todos votam. Se 5 ou mais concordarem, temos consenso.
6. Se não houver consenso:

   * A Suprema Juíza entra em ação, sendo treinada com base apenas no input e na predição da Juíza.
   * Itera várias vezes até convergir ou atingir o limite de ciclos.
7. A advogada atualiza-se com a opinião da Suprema Juíza.
8. O ciclo se repete até alcançar consenso total ou o limite de iterações.

## 🎓 Inspiração

Inspirado em sistemas democráticos e deliberativos, o D3E busca resolver tarefas ambíguas promovendo o debate interno entre modelos. Essa abordagem estática e iterativa é especialmente eficaz em tarefas onde a solução correta é incerta.

## 🔢 Características

* Controle de consenso via `tol` (threshold entre 0.6 a 0.98)
* Ciclos iterativos com máximo de 150 iterações por tarefa
* Visualização por heatmaps dos votos e consenso
* Deliberação com feedback realimentado
* Sistema modulado por simulações neurais (SimuV1-V5)

## 🔧 Execução

```bash
python main.py --mode test_challenge
```

## 🎥 Saídas

* `votos_visuais/`: imagens de predição por iteração
* `predictions_test/`: arquivos de predição finais por desafio

## 🔍 Futuras melhorias

* Otimização de ciclos via early stopping
* Juradas com ruído negativo regulador (adversarial noise)
* Variações inspiradas em jurados dissidentes

## 🚀 Contribuição

Pull requests e melhorias são bem-vindas. O tribunal está sempre aberto a novas vozes.

---

Copyright © 2025
