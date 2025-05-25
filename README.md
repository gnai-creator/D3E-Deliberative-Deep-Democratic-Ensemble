# 🧠 D3E — Deliberative Deep Democratic Ensemble

![D3E Logo](A_logo_for_D3E,_"Deliberative_Deep_Democratic_Ense.png)

## 📚 Overview
D3E simulates a deliberative courtroom of neural agents — jurors, lawyers, judges, and a supreme judge — each with their own role in collectively resolving ambiguous perception tasks from the ARC Challenge.

This system prioritizes process over speed, engaging in iterative voting and refinement until consensus is reached. Each agent is an independently trained neural network that evolves through simulated deliberation.

## ⚙️ System Hierarchy
- **Juradas (x3)**: Learn from the Advogada, add plurality of views.
- **Advogada**: Learns from the Suprema Juíza.
- **Juíza**: Aggregates juror and lawyer outputs.
- **Suprema Juíza**: Trains on raw data, acts as the final authority.

## 🌀 Deliberation Cycle
1. Raw input → Advogada → Juradas
2. Juíza aggregates → Suprema Juíza refines
3. If consensus (≥5/6) not reached → repeat cycle
4. Advogada & Suprema Juíza always train from raw input

## 🧪 Why This Works
- Promotes interpretability through stepwise visualization
- Resilient to noise and ambiguity
- Models improve each iteration

## 🧪 Results
Performance varies per challenge:
- 🟢 Task `00576224`: ✅ Consensus reached
- 🟡 Task `007bbfb7`: ⚠️ Partial agreement
- 🔴 Task `009d5c81`: ❌ Failed under strict consensus threshold

## 🔍 Insights
- High tolerance (`tol`) leads to slower but more accurate consensus
- `MAX_CYCLES` and `EPOCHS` balance speed vs convergence
- Juror disagreement can be intentionally injected for robustness

## 🔧 Files
- `court_logic.py` — Core deliberative logic
- `main.py` — Execution script
- `metrics_utils.py` — Visual voting logs and metrics
- `SimuV1-V5.py` — Neural models

## 📦 Dependencies
```bash
pip install tensorflow numpy seaborn matplotlib opencv-python
```

## 🧠 Inspirations
- Legal systems
- Democratic voting
- Human group deliberation

## 📺 Demos
- [ARC 00576224 Demo](https://youtu.be/eXO_PCb6M6E)
- [ARC 007bbfb7 Demo](https://youtu.be/0L6NJQhLlxE)

## 🤖 Final Thoughts
This is not just ensemble learning — it’s ensemble **deliberation**.

> "Consensus isn’t given, it’s earned. One vote at a time."

---
© 2025 Maya Rahto. Court is now in session.
