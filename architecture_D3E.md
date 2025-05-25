# D3E Architecture Specification

This document provides a technical breakdown of the architecture behind the Deliberative Deep Democratic Ensemble (D3E), detailing the role, function, and interfaces of each model component within the system.

---

## 🧠 Institutional Roles

### 1. Jurors
- **Type**: Independent neural models (e.g., CNNs)
- **Function**: Generate base predictions for a given input.
- **Input shape**: `(1, 30, 30, 1, 4)`
- **Output shape**: `(1, 30, 30)` (binary class logits or probabilities)

### 2. Lawyer (SimuV4)
- **Type**: Aggregator model
- **Function**: Synthesizes juror outputs into an intermediate representation
- **Input shape**: `(1, 30, 30, 1, 4)`
- **Output shape**: `(1, 30, 30)`
- **Training target**: Consensus mask from jurors

### 3. Judge (SimuV5)
- **Type**: Decision-making model
- **Function**: Learns patterns in the lawyer's synthesis and produces refined verdicts
- **Input shape**: `(1, 30, 30, 1, 40)`
- **Output shape**: `(1, 30, 30)`

### 4. Supreme Judge
- **Type**: Meta-evaluator
- **Function**: Verifies verdict quality using confidence metrics (accuracy, loss)
- **Decision criteria**:
  - `loss < 0.001`
  - `accuracy == 1.0`
  - Optional: detector confidence or anomaly scoring

---

## ⚙️ Internal Functions

### `add_judge_channel()`
Adds an additional dimension to a tensor with duplicated judgment or confidence channels. Used to prepare inputs for Lawyer and Judge.

---

## 🔄 Deliberation Process

1. **Juror Voting**: Each model votes independently.
2. **Aggregation**: Lawyer converts votes into learned intermediate representation.
3. **Refinement**: Judge processes this tensor into a final proposal.
4. **Evaluation**: Supreme Judge checks if proposal meets confidence threshold.
5. **Repetition**: If rejected, process restarts.

---

## 🧪 Confidence Evaluation

Each stage includes confidence metrics:
- **Consensus score** (across jurors)
- **Agreement level** (via accuracy/loss comparison)
- **Internal detector** (simple binary classifier trained on good vs. rejected masks)

---

## 📦 Integration Interfaces

### Module I/O Summary
| Component     | Input Shape               | Output Shape            |
|---------------|----------------------------|--------------------------|
| Jurors        | `(1, 30, 30, 1, 4)`        | `(1, 30, 30)`           |
| Lawyer (V4)   | `(1, 30, 30, 1, 4)`        | `(1, 30, 30)`           |
| Judge (V5)    | `(1, 30, 30, 1, 40)`       | `(1, 30, 30)`           |
| Supreme Judge | `(1, 30, 30)` + metrics    | `accept/reject`         |

---

## 🌐 Federation Architecture (Optional)

For multi-agent systems:
- Each robot has its own D3E instance
- Verdicts + confidences can be sent to central coordinator via REST/ROS
- Central node may:
  - log global consensus rates
  - override individual outcomes if quorum is broken

---

## 🛠️ Example API
```json
POST /api/deliberate
{
  "task_id": "001",
  "input": [[[[...]]]],
  "robot_id": "R2D3",
  "consensus_level": 0.88,
  "verdict": [[...]]
}
```

---

## 📚 Recommendations
- Log all intermediate decisions
- Include timestamps and robot ID for traceability
- Store consensus failures for further training of the Supreme Judge

---

## 🔮 Future Enhancements
- Lawyer LLM for natural-language explanation of consensus
- Visual heatmaps of disagreements
- Online learning for all roles
- Swarm jurisprudence: case law across robots

---

## Author
Felipe Maya Muniz — Independent Researcher