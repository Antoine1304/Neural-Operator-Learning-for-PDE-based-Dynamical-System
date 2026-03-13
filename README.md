# Neural Operator Learning for PDE-based Dynamical System

This repository contains the source code, datasets, and analysis of my final project for the "AI in the Sciences and Engineering" course at ETH Zurich, for which I achieved a perfect grade of 6/6.

The objective of this project is to explore, implement, and optimize state-of-the-art Deep Learning architectures for solving Partial Differential Equations (PDEs) and modeling continuous dynamical systems.

## Project Structure

The repository is modularly organized to separate data, source code, and result analysis:

```text
├── assets/                 # Generated visualizations organized by model
│   ├── PINNs/              
│   ├── FNOs/               
│   └── GAOT/               
├── datasets/               # Training, validation, and testing datasets (.npy files)
├── docs/                   # Final project report detailing mathematical analysis and results
├── src/                    # PyTorch source codes for the 3 main tasks     
└── README.md               # This file

-- Main Achievements and Architectures --
This project is divided into three main research directions:

1. Physics-Informed Neural Networks (PINNs) & Optimization Visualization
Solving a multi-scale Poisson equation by comparing a purely supervised approach (Data-Driven) and a PINN approach.

Curriculum Training: Implementation of a progressive training scheme to overcome convergence failures of PINNs on high-frequency targets (K=16).

Loss Landscapes: Generation and 3D visualization of the optimization space around local minima to analyze the stiffness and complexity of physics-informed gradients.

2. Fourier Neural Operators (FNO) and Transfer Learning
Training an FNO to approximate the evolution of an unknown dynamical system over time.

Implementation of spectral convolutions via FFT for one-to-one and all-to-all mappings.

Transfer Learning: Demonstration of the model's adaptability to a distribution shift in initial conditions. By fine-tuning on only 32 trajectories, the relative L2 error was drastically reduced from 15.85% (zero-shot) to 11.75%.

3. Geometry-Aware Operator Transformer (GAOT)
Extending the classic GAOT architecture to make it robust to irregular geometries.

Random Sampling & Dynamic Radius: Replacing the structured grid tokenization with random spatial sampling using a dynamic aggregation radius based on local density (inspired by RIGNO) via KD-Tree queries.

Positional Encoding & Perceiver: Implementation of continuous relative biases (CRB) and a Cross-Attention compression mechanism (Perceiver). This allowed processing 1024 latent tokens while maintaining a competitive L1 error (16.44%) and reducing computational cost.

-- Technologies Used --
Language: Python

Deep Learning: PyTorch (Autograd, spectral models, Transformers, Adam/L-BFGS hybrid optimization)

Numerical Computation & Graphs: NumPy, SciPy (cKDTree for dynamic neighborhoods)

Visualization: Matplotlib (3D surface plots, wave animations)

-- Detailed Report --
For an in-depth analysis of error metrics, hyperparameter configurations, and physical observations, please refer to the complete project report (PDF) located in the docs folder.