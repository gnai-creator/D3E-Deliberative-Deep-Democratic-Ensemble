# 🧠 SimuV1 ARC Judiciary System

**"If the human mind can deliberate, so can a swarm of pixel-obsessed neural networks."**

SimuV1 é uma arquitetura visual neural projetada para o Abstraction and Reasoning Corpus (ARC). Após resolver mais de **205 desafios de ARC-AGI-2**, evoluímos para uma nova abordagem inspirada em sistemas judiciais, votações e metacognição.  
Bem-vindo ao **ARC Judiciary System**.

## 🧬 Estrutura do Sistema

O sistema é composto por **5 redes neurais distintas** com papéis específicos:

| Rede | Papel | Descrição |
|------|-------|-----------|
| **IA1–IA3** | Juradas | Continuam treinando com base nas previsões da IA4. Cada uma possui variações arquiteturais ou de dados. |
| **IA4** | Advogada | Gera previsões iniciais baseadas apenas nos inputs de teste. |
| **IA5** | Juíza | Avalia os outputs de IA1–IA3 com base em distância semântica e um campo especial chamado `juízo`. Decide qual output é o mais confiável. |

## 🌀 O Ciclo

1. Treinamento inicial com dados de treino (train[0]["input"] e train[0]["output"]).
2. IA4 realiza **inference** no conjunto de testes.
3. Suas previsões alimentam o treinamento supervisionado de IA1–IA3.
4. IA1–IA3 devolvem previsões + confiança (`juízo ∈ [0, 1]`).
5. IA5 avalia as previsões usando critérios de consenso e confiança.
6. Se pelo menos 3 outputs possuem `juízo ≥ 0.9`, o voto é aceito.
7. Votação final entre as 5 redes define a resposta.

## 🧠 Sobre o Juízo

A dimensão `juízo` é um campo contínuo que representa a autoconfiança da IA sobre sua própria resposta.  
Ele é aprendido durante o treinamento via uma *critic head* que tenta prever a loss esperada.

## 🎯 Objetivo

- Resolver **todas as 400 tarefas do ARC-AGI-2**.
- Sem vazamento de dados.
- Apenas inferência legítima com generalização.

## 📼 Registro

Todo o processo foi registrado em vídeo:
- SimuV1 resolvendo 205 tarefas: [YouTube Link](https://www.youtube.com/watch?v=o3It0tT4kGk)

## ⚖️ Filosofia

> "Você não programa esse sistema. Você educa ele."

Este projeto trata mais de criar uma **mente deliberativa coletiva** do que um simples modelo de classificação. É um passo em direção à inteligência artificial interpretável, cooperativa e autocrítica.

---

**Status:** Em desenvolvimento contínuo.  
**Contato:** via GitHub Issues ou ~visões neurais telepáticas~ futuras releases.

