# ARC Challenge - Paradigms for Model Training and Debate Evaluation

This project explores two distinct paradigms for training and deploying models to solve tasks in the Abstraction and Reasoning Corpus (ARC) challenge. Both approaches use ensembles of neural models evaluated through a debate-style consensus mechanism, but they differ in their architectural diversity and specialization strategies.

## Overview

The objective is to learn from limited examples in a grid-based transformation problem space. Models are trained on few-shot input/output pairs and are later evaluated through multiple rounds of "discussion" (inference voting) to determine consensus outputs.

## Paradigm 1: Homogeneous Ensemble (Clone Strategy)

In this setup, all models in the ensemble are instances of the same architecture (e.g., `SageAxiom`). This promotes redundancy, stability, and consistency during training and evaluation. Each model is trained independently with slight variation due to stochastic optimization.

### Pros:

* Consistent behavior across models.
* Easier to interpret performance discrepancies.
* Uniform model complexity and resource use.

### Cons:

* Limited architectural diversity may result in blind spots.
* All models may share the same biases and weaknesses.

## Paradigm 2: Heterogeneous Ensemble (Specialized Strategy)

In this approach, the ensemble is composed of models with different architectural specializations:

* A complete model (`SageAxiom`) with all modules enabled.
* Variants with subsets of the architecture (e.g., reduced attention, no memory, simplified refinement).

This fosters complementary strengths, enabling more diverse problem-solving heuristics within the ensemble.

### Pros:

* Architectural diversity can lead to better generalization.
* Voting debates benefit from different reasoning styles.
* Improves robustness to task variation.

### Cons:

* More complex to manage and debug.
* Performance attribution becomes harder.
* Requires careful balancing to avoid model redundancy or overfitting.

## Debate Mechanism

Each task is evaluated by running the trained ensemble in a multi-round voting system. The output selected is the one that receives a majority consensus among models. If no consensus is reached, fallback strategies can be employed.

## Results & Tracking

Each training and evaluation session logs individual model outputs, round history, vote counts, and similarity scores. This enables meta-analysis of ensemble behavior and helps identify the most influential or accurate models within a debate cycle.

## Conclusion

Both paradigms offer distinct benefits. The homogeneous strategy ensures consistency and controlled testing conditions, while the heterogeneous strategy enhances diversity and coverage. Together, they provide a comprehensive framework to explore automated reasoning under few-shot learning constraints.
