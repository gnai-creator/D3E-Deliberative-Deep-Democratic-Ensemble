# D3D System Architecture

The **D3D (Deliberative Deep Democratic Ensemble)** architecture is a symbolic and neural hybrid framework that models a judicial-like decision system using deep learning agents with distinct roles, weights, and consensus mechanisms.

---

## 🔄 High-Level Flow

```
           +---------------------+
           |   Input Task (X)   |
           +---------------------+
                      |
        +-------------+-------------+
        |                           |
+----------------+       +---------------------+
|  Jurors (0-2)  | ...   |   Lawyer (Model 3)   |
+----------------+       +---------------------+
        |                           |
        +-------------+-------------+
                      |
                +-----------+
                |   Judge   | (Model 4)
                +-----------+
                      |
          +-----------+-----------+
          |                       |
+---------------------+  +---------------------+
| Supreme Judge (5)   |  |  Prosecutor (6)     |
| (Retrains per round)|  | (Creates antithesis)|
+---------------------+  +---------------------+
          |                       |
          +-----------+-----------+
                      |
                +------------+
                | Consensus  |
                | Evaluation |
                +------------+
                      |
                +------------+
                |   Output   |
                +------------+
```

---

## 🪧 Component Roles

| Role              | Model Index | Description                                               |
| ----------------- | ----------- | --------------------------------------------------------- |
| **Jurors**        | 0-2         | Provide diverse initial predictions                       |
| **Lawyer**        | 3           | Intermediate mediator, higher influence                   |
| **Judge**         | 4           | Core decider with more authority                          |
| **Supreme Judge** | 5           | Retrains iteratively using provisional consensus as label |
| **Prosecutor**    | 6           | Challenges the consensus with an antithesis               |

---

## 🧳️ Deliberation Cycle

1. **Prediction**: Each model produces a symbolic prediction on `X_test`.
2. **Filtering**: Predictions are filtered to keep only valid class values.
3. **Aggregation**: Pixel-wise mode of predictions is used to form a provisional consensus.
4. **Training**: Supreme Judge is retrained with this consensus as supervision.
5. **Antithesis**: Prosecutor generates opposing prediction to force refinement.
6. **Confidence Voting**: Weighted consensus is computed using the confidence system.
7. **Exit Condition**: If consensus exceeds tolerance threshold, exit loop; otherwise continue.

---

## 🚀 Consensus and Confidence System

* Each model has a **weight** and **confidence level**.
* A **confidence manager** tracks performance and adjusts voting influence.
* **Consensus** is computed as a weighted agreement score.

---

## 📁 Files Overview

| File                   | Responsibility                                        |
| ---------------------- | ----------------------------------------------------- |
| `GrampyX.py`           | Main class managing training, inference, saving state |
| `court_logic.py`       | Core of the deliberation loop and symbolic consensus  |
| `confidence_system.py` | Manages weights and trust/confidence of each model    |
| `model_compile.py`     | Compilation and definition of models                  |
| `metrics.py`           | Custom symbolic loss and metrics for evaluation       |
| `data_pipeline.py`     | Loads, normalizes, and structures task data           |

---

## 🚫 Key Principles

* No single model dominates: balance between cooperation (Supreme Judge) and adversariality (Prosecutor)
* Symbolic reasoning is prioritized over raw accuracy
* Models evolve dynamically with exposure to consensus vs antithesis tension

---

## 🌍 Applications Beyond ARC

* Ethical AI alignment
* Symbolic planning
* Ensemble-based anomaly detection
* Adversarial robustness research

---

> D3D reimagines learning as a symbolic court — deliberative, democratic, and dynamic.
