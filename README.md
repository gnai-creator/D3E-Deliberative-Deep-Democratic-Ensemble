# D3D: Deliberative Deep Democratic Ensemble

**ARC Court Logic** is a symbolic multi-agent deliberation framework designed for solving ARC-style visual reasoning tasks. It simulates a hierarchical judicial system using deep learning agents with cooperative and adversarial roles.

## 🌐 Overview

ARC Court Logic uses an ensemble of models acting as jurors, lawyers, judges, and a prosecutor. Each model has a defined hierarchical weight and confidence level, contributing to a symbolic voting system. Deliberation is conducted in iterative cycles until consensus is reached or a maximum number of iterations is exceeded.

## 🧑‍🏫 Roles and Responsibilities

| Model Index | Role          | Behavior                                               |
| ----------- | ------------- | ------------------------------------------------------ |
| 0-2         | Jurors        | Initial predictive agents                              |
| 3           | Lawyer        | Intermediate predictor with higher weight              |
| 4           | Judge         | Authoritative predictor                                |
| 5           | Supreme Judge | Aggregates and retrains based on consensus             |
| 6           | Prosecutor    | Generates antithesis prediction to challenge consensus |

## ⚖️ Deliberation Process

1. **Prediction Generation**: Each model makes an initial prediction on the test grid.
2. **Filtering**: Predictions are filtered to keep only valid classes.
3. **Consensus Building**: A symbolic majority vote (pixel-wise mode) determines the provisional consensus.
4. **Supreme Court Training**: The Supreme Judge is trained on the current consensus.
5. **Prosecutor Opposition**: A counter-prediction is generated and used to challenge the consensus.
6. **Final Evaluation**: Weighted consensus is computed. If above threshold, the decision is accepted.

## ⚙️ Architecture

* TensorFlow-based Keras models for jurors and judges
* Dynamic consensus management using `ConfidenceManager`
* Persistent state saving via Pickle and HDF5
* Visual and symbolic debugging tools for interpretability

## 🔮 Key Files

| File                   | Description                              |
| ---------------------- | ---------------------------------------- |
| `GrampyX.py`           | Main entry point and model manager       |
| `court_logic.py`       | Core deliberation logic                  |
| `confidence_system.py` | Consensus evaluation and trust weighting |
| `model_compile.py`     | Model compilation and loss definitions   |
| `metrics.py`           | Custom symbolic evaluation metrics       |
| `SimuV1.py`            | Simulation runner for batch testing      |

## 🔄 Training & Execution

Run the training and evaluation loop:

```bash
python main.py
```

## 📈 Evaluation

The system saves symbolic prediction plots and records consensus confidence. Final predictions are exported in a JSON submission file.

## 🚀 Future Work

* Add more sophisticated prosecutor behavior
* Integrate self-refinement in jurors
* Extend to new ARC-like benchmarks

## ✅ License

CC BY-ND 4.0
