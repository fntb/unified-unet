# Unified U-Net

This repository contains an **algorithmic and experimental study** of the paper  
*A Unified Framework for U-Net Design and Analysis*  
by Williams, Falck et al.

The project was carried out in the context of a Master-level research project and focuses on **understanding, implementing, and validating** the core ideas of the unified U-Net framework through controlled experiments.

> Status: work in progress, but experimental results are stable and documented.

---

## Project Scope

The goal of this repository is **not** to propose a new architecture, but to:
- reformulate U-Net architectures using the unified operator-based framework introduced in the paper,
- study the role of **multiscale structure**, **residual learning**, and **preconditioning**,
- validate these ideas experimentally on **synthetic signals** and **standard image datasets**.

The emphasis is placed on **interpretability and controlled experimentation**, rather than large-scale benchmarking.

---

## What is implemented

### Architectures

- **Baseline U-Net**
  - classical encoder–decoder architecture
  - used as a reference model

- **Residual formulation**
  - the network learns a correction to the input signal
  - connects U-Nets to residual networks and operator learning

- **Preconditioned formulation**
  - introduction of an explicit approximation operator $( P )$
  - reconstruction written as:  
    $$[\hat{x} = P(y) + f_\theta(y)]$$
  - interpretation aligned with the unified U-Net framework

- **Haar wavelet preconditioning**
  - fixed (non-learnable) Haar-based multiscale approximation
  - explicit separation between coarse approximation and learned correction
  - direct link with multiresolution analysis discussed in the paper

---

### Data and tasks

- **Synthetic 1D signals**
  - controlled construction of signals combining:
    - smooth low-frequency components,
    - intermediate oscillations,
    - localized events (spikes),
  - supervised denoising task,
  - frequency-based error analysis (low vs high frequencies).

- **Image datasets**
  - MNIST (classification / reconstruction)
  - Kvasir (medical image segmentation)

These datasets are used to verify that the observed mechanisms generalize beyond toy examples.

---

## Experimental focus

The experiments are designed to isolate and study:
- the effect of residual learning,
- the impact of explicit preconditioning,
- the role of multiscale structure induced by Haar decompositions.

Evaluation includes:
- training and validation dynamics,
- reconstruction quality,
- frequency-domain error analysis (for synthetic signals).

All experiments are run with **controlled settings and fixed random seeds** to ensure reproducibility.

---

## Technical stack

- Python
- PyTorch
- PyTorch Lightning
- Hydra
- `uv` for environment and dependency management

---

## Quick start

Run an experiment using:

```bash
uv run src/main.py
