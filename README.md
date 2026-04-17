# Optimization of Fock-State Preparation in a Hybrid Quantum System using a Neural-Network Surrogate Model

This repository contains a side project that emerged from my master's thesis work on the optimization of state preparation in hybrid quantum systems. A neural network is trained on precomputed data and then used as a surrogate model to optimize the control parameters of a quantum dynamical system.

## Project idea

The workflow is intentionally compact and research-oriented:

- a dense neural network is trained to approximate the target-state fidelity,
- the trained surrogate is then used as a fast optimizer for the control parameters
- the optimal parameters found via the surrogate model are evaluated using a benchmark model that computes the exact dynamics

This repository currently focuses on the case $N=1$ with three pulses and fixed nonlinear phases. Because the phases are fixed, the surrogate only needs five effective input parameters instead of the full pulse vector.

For a more detailed description of the physics and the surrogate-model setup, see [docs/physics_and_approach.md](docs/physics_and_approach.md).

## Key findings

The current experiments show that the chosen architecture learns the fidelity landscape induced by the exact dynamics very well. The stored validation metrics indicate an average exact-surrogate difference of about $0.008$.

An especially encouraging result is that the training pipeline removes samples above a fidelity threshold of $0.9$, yet the surrogate-guided gradient optimization still finds solutions with exact fidelities of about $0.97$. This suggests that surrogate-based methods may remain useful for more complicated pulse sequences and higher-dimensional optimization problems.

## Repository structure

### [Dynamics_with_unitary_operators.py](Dynamics_with_unitary_operators.py)
This script contains the exact physics-based benchmark model.

It:
- builds the operator matrices,
- defines the Hamiltonians,
- computes the unitary dynamics of the three-pulse sequence,
- evaluates the target-state probability for a given parameter set.

### [Training.py](Training.py)
This script trains the surrogate neural network.

It:
- loads the precomputed dynamics data,
- extracts the relevant input features,
- computes labels based on the target-state fidelity,
- trains a dense neural network to approximate the physical benchmark,
- saves the trained surrogate model.

### [Surrogate_Model_GD.py](Surrogate_Model_GD.py)
This script performs optimization using the trained surrogate model.

It:
- loads the trained network,
- computes gradients of the model output with respect to the input parameters,
- performs multiple Adam-based optimization runs from random initial points,
- validates the optimized parameters against the exact dynamical model,
- visualizes the comparison between neural-network predictions and true physical values.

## Data and artifacts

- [Three_Pulse_Dynamics_Data.npy](Three_Pulse_Dynamics_Data.npy): dataset generated from the physical benchmark
- [random_init_params.npy](random_init_params.npy): random initializations for optimization runs
- trained model artifacts are created during training and reused for surrogate-based optimization

## Setup

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Typical workflow

1. Generate or provide the dynamics dataset.
2. Train the surrogate model with [Training.py](Training.py).
3. Run surrogate-based optimization with [Surrogate_Model_GD.py](Surrogate_Model_GD.py).
4. Compare neural-network predictions against the exact benchmark.

## Example results

The repository also contains example output figures produced during training and surrogate-guided optimization.

### Training history

![Training history](artifacts/figures/training_history.png)

### Validation: exact vs. surrogate

![Validation comparison](artifacts/figures/validation_exact_vs_surrogate.png)

### Surrogate-guided optimization outcome

![Optimization comparison](artifacts/figures/gd_fidelity_comparison.png)

## Motivation and outlook

This project is intended as a compact demonstration of how machine learning can support quantum-control problems in hybrid quantum systems. Even in the reduced three-pulse setting, the surrogate model captures the benchmark faithfully and can guide the search toward near-unity fidelities.

A natural next step would be to extend the framework to more demanding pulse structures, such as four-pulse protocols or longer sequences, and to include dissipative effects or loss. Those are precisely the scenarios in which surrogate-based optimization becomes most attractive, since the cost of repeated exact simulations grows quickly.

## Notes

The code is intentionally kept close to the original research workflow. The focus of the repository is on clarity, physical interpretability, and demonstrating the combined use of simulation and machine learning in a compact project.
