# SimuV1: A Cortex-Inspired Pattern Resolver

> SimuV1 (short for "Simulated Visual Cortex V1") is a neural architecture inspired by the early layers of biological vision, adapted to tackle the ARC Challenge like it's a series of sacred mosaics.

Built for grid-based reasoning. Forged in the fire of convolutional obsession. Forget image classification — this thing is here to understand abstract visual logic like a nervous system with a grudge.

## What is SimuV1?

SimuV1 is a pattern abstraction and replication engine tailored to solve ARC (Abstraction and Reasoning Corpus) tasks. It is a specialized visual processor that learns how to "see" logic — tiling, symmetry, transformations — with minimal data.

It does not generalize across tasks. It conquers each one individually, like a slightly deranged craftsman reinventing the wheel 125 times and making it sharper every time.

## Features

* **Fractal Convolutional Backbone**
  Multi-scale `FractalBlock` modules for expressive pattern encoding.

* **Learned Color Permutations**
  Because ARC loves arbitrary palette shuffles and SimuV1 eats those for breakfast.

* **Symmetry-aware Preprocessing**
  With `LearnedFlip` and `DiscreteRotation`, the model learns what to ignore and what to mirror.

* **Spatial Attention Over Learned Memory**
  A tiny, angry transformer core that yells "FOCUS!" at the visual field.

* **Presence Head**
  Predicts where objects should exist. Not unlike the part of your brain that says “that’s weird” when a thing disappears.

* **Grid Input / Grid Output**
  Everything’s a grid. Inputs, outputs, internal neuroses. And yes, it assumes you like padding.

## Usage

```python
from shape_locator_net import SimuV1
from model_compile import compile_model

model = SimuV1(hidden_dim=256)
model = compile_model(model, lr=1e-3)

model.fit(x_train, y_train, epochs=60, ...)
preds = model(x_test, training=False)
```

## Performance

* Solves over **124 out of 125 ARC test tasks**
* Typical task convergence in **5–20 training cycles**
* Collapses gracefully when exposed to unexpected grid shapes or reality

## Limitations

* **No inter-task generalization** — each task is learned from scratch with fresh weights, just like your goldfish when it forgets it already swam past the castle.
* **Shape-specific** — SimuV1 expects well-behaved inputs, and will politely explode otherwise.
* **Not a general AI** — but it plays one surprisingly well.

## Philosophy

> SimuV1 doesn’t think. It **recognizes**.
> It doesn’t understand. It **recursively adapts**.
> It doesn’t generalize. It **perfects each microcosm, one tiled hallucination at a time**.

## Naming Notes

SimuV1 is named after:

* **Simulated Visual Cortex V1** — the real-world module responsible for early-stage visual processing.
* The **simplicity** and **rigidity** of early perception.
* And because “CortexMcGridFace” didn’t make it past the branding meeting.
