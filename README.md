# Neural Operator Learning for PDE-Based Dynamical Systems

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![License](https://img.shields.io/badge/license-MIT-green)

## Motivation

Traditional numerical solvers for complex physical systems (such as fluid mechanics and Navier-Stokes equations) are notoriously computationally expensive and slow. Driven by a deep interest in accelerating these simulations, I developed this project to explore and optimize state-of-the-art Deep Learning architectures for solving Partial Differential Equations (PDEs) and modeling continuous dynamical systems.

[cite_start]This repository contains the source code, datasets, and analysis of my research into Physics-Informed Neural Networks (PINNs), Fourier Neural Operators (FNOs), and Geometry-Aware Operator Transformers (GAOT)[cite: 11].

## Key Architectures & Achievements

### 1. Physics-Informed Neural Networks (PINNs) & Loss Landscapes
[cite_start]Solving a multi-scale Poisson equation by comparing a purely supervised Data-Driven approach with a PINN approach[cite: 11, 313, 324].
* [cite_start]**Curriculum Training:** Implemented a progressive training scheme to overcome PINN convergence failures on high-frequency targets ($K=16$)[cite: 329, 338, 341].
* [cite_start]**Loss Landscapes Visualization:** Generated 3D visualizations of the optimization space around local minima to analyze the stiffness and complexity of physics-informed gradients[cite: 466, 504].

![PINN Loss Landscape](assets/PINNs/loss_landscape_placeholder.png)
*> Figure 1: 3D Loss landscapes comparing Data-Driven and PINN models across different spectral complexities. (Replace with Figure 4 from your report)*

### 2. Fourier Neural Operators (FNO) and Transfer Learning
Training an FNO to approximate the evolution of an unknown dynamical system over time.
* [cite_start]**Spectral Convolutions:** Implemented via FFT for one-to-one and all-to-all mappings[cite: 517, 579, 580].
* **Transfer Learning:** Demonstrated the model's adaptability to a distribution shift in initial conditions. [cite_start]By fine-tuning on only 32 trajectories, the relative $L_2$ error was drastically reduced from 15.85% (zero-shot) to 11.75%[cite: 667, 668].

![FNO Prediction](assets/FNOs/fno_prediction_placeholder.png)
*> Figure 2: Ground Truth solution vs all-2-all finetuned model prediction for an unknown initial distribution at t=1s. (Replace with Figure 10 from your report)*

### 3. Geometry-Aware Operator Transformer (GAOT)
Extending the classic GAOT architecture to make it robust to irregular geometries.
* [cite_start]**Random Sampling & Dynamic Radius:** Replaced the structured grid tokenization with random spatial sampling using a dynamic aggregation radius based on local density (inspired by RIGNO) via KD-Tree queries[cite: 798, 861].
* [cite_start]**Positional Encoding & Perceiver:** Implemented continuous relative biases (CRB) and a Cross-Attention compression mechanism (Perceiver)[cite: 895, 896, 918]. [cite_start]This allowed processing 1024 latent tokens while maintaining a competitive $L_1$ error (16.44%) and reducing computational cost[cite: 925].

![GAOT Coverage](assets/GAOT/gaot_coverage_placeholder.png)
*> Figure 3: Number of latent tokens covering every point of the grid using the dynamic radius strategy. (Replace with Figure 14 from your report)*

---

## Project Structure

```text
├── assets/                 # Generated visualizations organized by model
│   ├── PINNs/              
│   ├── FNOs/               
│   └── GAOT/               
├── datasets/               # Training, validation, and testing datasets (.npy files)
├── docs/                   # Detailed project report (mathematical analysis and results)
├── src/                    # PyTorch source codes for the 3 main tasks     
│   ├── run_pinn.py         # (Adapt these filenames to your actual scripts)
│   ├── run_fno.py          
│   └── run_gaot.py         
└── README.md