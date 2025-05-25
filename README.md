![Court System](Court%20System.png)

# ARC Supreme Court: Deliberative Ensemble for ARC Challenges

This repository contains a legal-inspired ensemble learning system designed to tackle the ARC (Abstraction and Reasoning Corpus) challenge using a courtroom metaphor.

## Overview

The system consists of multiple neural models, each playing a role in a deliberative process:

* **Juradas (Jurors)**: 3 models that initially learn from the advogada (lawyer).
* **Advogada (Lawyer)**: Learns directly from raw input and later from the Supreme Judge.
* **Juíza (Ordinary Judge)**: Aggregates the predictions from the jurors and the lawyer to issue a decision.
* **Suprema Juíza (Supreme Judge)**: Trains on raw input using the decision from the ordinary judge as supervision, and refines predictions until a consensus is reached.

This structure simulates a deliberative judicial system where models improve over iterations, refining their collective decision.

## Features

* **Deliberative Voting**: All members vote on the output. Consensus is required from 5 out of 6 models to finalize a solution.
* **Supreme Justice Refinement**: The Suprema Juíza iteratively improves until reaching accuracy or loss thresholds.
* **Iterative Learning**: Jurors and judges learn from one another in successive iterations, simulating a learning ecosystem.
* **Visual Output**: Saves vote visualizations per iteration for inspection.

## System Flow

```
Raw Input → Lawyer & Supreme Judge
           → Jurors (trained from lawyer)
           → Ordinary Judge aggregates
           → Supreme Judge trains using judge's output
           → Lawyer retrains from Supreme Judge
Repeat until consensus ≥ threshold
```

## File Structure

* `main.py`: Entry point for training and testing on ARC tasks.
* `court_logic.py`: Core logic simulating the court's deliberation process.
* `metrics_utils.py`: Visualization and evaluation utilities.
* `model_loader.py`, `model_compile.py`: Utilities for initializing and compiling models.
* `SimuV1.py` to `SimuV5.py`: Model architecture definitions.
* `neural_blocks.py`: Shared blocks used by model architectures.
* `data_preparation.py`: Dataset loading and preprocessing utilities.

## Running the System

Make sure you have the necessary dependencies:

```bash
pip install tensorflow matplotlib seaborn pandas opencv-python
```

To run an ARC task:

```bash
python main.py --task_id <task_id>
```

This will generate logs, save vote visualizations in `votos_visuais/`, and optionally export a video of the deliberation process.

## Notes

* The system is inspired by legal structure but remains entirely a computational model.
* It is not optimized for all ARC tasks; it performs best in ambiguous scenarios where iterative consensus can be effective.

## Videos & Examples

Example demonstrations:

* [Deliberative ARC Challenge 1](https://youtu.be/eXO_PCb6M6E)
* [Challenge 2 Analysis](https://youtu.be/0L6NJQhLlxE)

## License

MIT. Use freely and judge wisely.

---

For academic inquiries or to file an appeal with the Supreme Juíza, contact the developer via GitHub Issues.
