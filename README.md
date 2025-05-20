# Sage Debate Loop: ARC Challenge Task-Specific Debate Framework

## Visão Geral

Este projeto implementa um pipeline completo de aprendizado de máquina para resolver desafios do tipo ARC (Abstraction and Reasoning Corpus) usando múltiplos modelos por tarefa. O diferencial é o uso de um "loop de debate" entre modelos para decidir uma resposta por maioria, inspirado em dinâmicas de consenso coletivo.

## Componentes Principais

### 1. `core.py`

Contém a definição da arquitetura principal `SageAxiom`, uma rede convolucional com atenção e rotação aprendida.

### 2. `metrics_utils.py`

Funções auxiliares para plotar histórico de treino, matriz de confusão, distribuição de logits e mapas de atenção.

### 3. `runtime_utils.py`

Funções utilitárias para logging, profile de tempo e padding de entradas.

### 4. `sage_debate_loop.py`

Loop de debate entre modelos. Em cada rodada, os modelos geram saídas e o sistema tenta atingir uma maioria de votos. Se houver maioria, a saída é considerada aceita.

### 5. `main.py`

Pipeline principal que:

* Carrega os desafios ARC
* Treina `MODELS_PER_TASK` modelos por tarefa
* Salva e recarrega os modelos por task
* Realiza inferência com debate
* Avalia desempenho em dados de validação e teste

## Como Funciona o Debate

* Cada task tem seus próprios modelos.
* Cada modelo gera uma saída para o mesmo input.
* Se 3 ou mais modelos concordarem (por padrão `WINNING_VOTES_COUNT = 3`), essa saída vence.
* Caso contrário, o loop prossegue até `max_rounds`.

## Requisitos

* Python 3.9+
* TensorFlow 2.11+
* `tensorflow-addons`
* `scikit-learn`
* `matplotlib`, `seaborn`

## Como Rodar

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Coloque seu arquivo de desafios no formato:

```json
{
  "task_id": {
    "train": [{"input": [[...]], "output": [[...]]}],
    "test": [{"input": [[...]]}]
  },
  ...
}
```

3. Execute:

```bash
python main.py
```

4. Saídas salvas em:

* `results/` (modelos por task)
* `submission.json` (respostas inferidas)
* `evaluation_logs.json` (histórico do debate)
* `images/` (gráficos de histórico e avaliação)

## Configurações

Modificáveis no `main.py`:

* `MODELS_PER_TASK`
* `EPOCHS`, `BATCH_SIZE`, `PATIENCE`
* `HOURS`, `MAX_TRAINING_TIME`, `MAX_EVAL_TIME`

## Contribuições Futuras

* Suporte a retreinamento incremental
* Salvar tempos por task (`task_times.json`)
* Visualização em tempo real do consenso

---

Se você chegou até aqui, parabéns. Este projeto está mais disciplinado que muita gente em grupo de TCC.
