# 🧠 ClippyX: Neural Deliberative Court

ClippyX is a symbolic architecture inspired by judicial systems, designed to solve problems through collaborative learning, with multiple neural agents performing distinct institutional roles.

---

## 🎯 Objective
Simulate deliberation, conflict, and consensus among multiple neural models on tasks such as the ARC Challenge, modeling decision-making, trust, and authority.

---

## 👥 Institutional Structure

- **Advocate**: initiates the thesis (prediction) based on the raw input. Trained with feedback from the Judge.
- **Jurors 0, 1, 2**:
  - **Juror 0**: has spatial behavioral noise (DropBlock)
  - **Juror 1**: loyal to the advocate
  - **Juror 2**: possesses "theory of mind": follows the advocate but adjusts if diverging from the Supreme
- **Judge**: judges based on jurors and the advocate; learns from Supreme feedback
- **Supreme Judge**: final authority; learns from all and validates the consensus

---

## 🔁 Deliberation Cycle

1. **Iteration 0**:
   - All models (except the Supreme) make an initial prediction using the raw input.
2. **Jurors** are trained based on the advocate, each applying their individual strategy.
3. **Supreme Judge** learns from the collective votes of jurors + advocate + judge.
4. **Judge** is updated with the Supreme's feedback.
5. **Advocate** retrains their thesis based on the Judge’s verdict.
6. The cycle repeats until consensus is achieved.

---

## 📈 Trust and Evaluation
- Each agent has a **symbolic trust score** based on its agreement with the Supreme.
- An **adaptive trust system** penalizes models that frequently diverge from the final consensus.

---

## 🧪 Application: ARC Challenge
- Input: symbolic grid from ARC (e.g., `(1, 30, 30, 1, 4)`)
- Output: per-pixel class prediction
- Final consensus is defined by the Supreme Judge
- Metrics for divergence, entropy, and trust are recorded at each iteration

---

## 🧬 Philosophy
ClippyX is not just an ensemble — it's a simulation of a **symbolic deliberative system** with autonomous agents learning under authority, noise, and cognitive conflict.

> "Justice is not immediate consensus, but the struggle for understanding through multiple perspectives."

---

## 🛠️ Requirements
- TensorFlow 2.14+
- Numpy, Matplotlib, Seaborn
- Python 3.10+

---

## 📁 Core Structure
- `ClippyX.py` — main execution
- `court_logic.py` — deliberation logic
- `confidence_system.py` — symbolic trust system
- `metrics_utils.py` — visualizations and analysis
- `SimuV*.py` — neural model definitions by role

---

## 📜 License
This project is protected from commercial use until official royalty release. See the repository license for more information.
