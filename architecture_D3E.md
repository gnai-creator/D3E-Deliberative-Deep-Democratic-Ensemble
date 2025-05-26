# 🧠 ClippyX Architecture – Neural Deliberation with Symbolic Hierarchy

ClippyX is an architecture inspired by judicial institutions, where neural models assume distinct social roles, engaging in hierarchical learning, deliberative cycles, and adaptive trust mechanisms.

---

## ⚖️ Overview

The architecture simulates a judicial system with six neural agents:

- 3 **Jurors**
- 1 **Advocate**
- 1 **Judge**
- 1 **Supreme Judge**

Each agent is a neural model trained with role-specific data during the deliberative cycle. Agents are not independent: they share context, interact through predictions and feedback, and adapt behavior based on authority and disagreement.

---

## 🧩 Institutional Roles

| Agent          | Role                                       | Initial Training                 | Adaptation                            |
|----------------|---------------------------------------------|----------------------------------|----------------------------------------|
| Juror 0        | Juror with behavioral noise                | Learns from Advocate + noise     | Does not adapt                         |
| Juror 1        | Loyal juror                                | Learns from Advocate             | Does not adapt                         |
| Juror 2        | Juror with Theory of Mind                  | Learns from Advocate             | Adapts to Supreme if divergent         |
| Advocate       | Starts the legal thesis                    | Learns from Judge's feedback     | Continuously adapts                    |
| Judge          | Weighs jurors and advocate                 | Learns from Supreme              | Re-evaluates based on Supreme's ruling |
| Supreme Judge  | Final authority                            | Learns from all others           | Defines final consensus                |

---

## 🔁 Deliberation Cycle

1. **Initial predictions (iteration 0)**:
   - All models (except the Supreme) predict based on symbolic input.
   - **All agents always receive the same raw symbolic input (`X`)**.
   - This `X` represents the raw state of the problem and remains constant across iterations.

2. **Training phase**:
   - Jurors learn from the Advocate.
     - Juror 0 applies spatial noise (DropBlock).
     - Juror 2 may be retrained based on the Supreme if divergence is detected.
   - The Supreme learns from all (jurors, advocate, judge).
   - The Judge is updated with the Supreme's feedback.
   - The Advocate refines their thesis with the Judge's feedback.

3. **Visualization and consensus analysis**:
   - Votes are visualized.
   - Entropy maps highlight pixel-level disagreement.
   - Trust system evaluates each model based on agreement with the Supreme.

4. **Repeat until consensus**:
   - If the Supreme achieves full accuracy on the Judge's output, the cycle ends.
   - Otherwise, a new iteration begins with updated votes.

---

## 📐 Symbolic Interaction Diagram

```text
Raw Input
   │
   ├──▶ Advocate
   │       │
   │       ├──▶ Juror 0 (noisy)
   │       ├──▶ Juror 1 (loyal)
   │       └──▶ Juror 2 (with ToM)
   │
   └──▶ Judge ◀── Supreme Judge
                     ▲
           ┌─────────┴──────────┐
           │                   │
     Jurors, Judge, Advocate ──┘
```

---

## 🎯 Advanced Features

- Modeling of **localized cognitive noise** (Juror 0)
- Simulation of **symbolic authority** (Supreme)
- Implementation of **adaptive Theory of Mind** (Juror 2)
- **Evolving symbolic trust system** with penalties
- Visualization of divergence using **symbolic entropy maps**

---

## 💡 Applications

- ARC Challenge benchmark
- Simulation of adaptive judgment
- Study of distributed trust in ensembles
- Symbolic metaphor for explainable AI

---

## 📜 License and Ownership
This system is protected against unauthorized commercial use. Royalties are required as described in the main license.
