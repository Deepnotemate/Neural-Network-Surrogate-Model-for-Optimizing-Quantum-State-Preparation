# Optimization of Fock-State Preparation in a Hybrid Quantum System using a Neural-Network Surrogate Model

This repository contains a side project that emerged from my master's thesis work on the optimization of state preparation in hybrid quantum systems.
Depending on external control pulse parameters, the dynamics of a hybrid quantum system can be steered such that the resulting final state has high fidelity with respect to a target Fock state. 

A neural network is trained on precomputed data to rerpesent the fidelity and then used as a surrogate model to optimize the control parameters of a quantum dynamical system.

For a more detailed description of the physics and the surrogate-model setup, see [docs/physics_and_approach.md](docs/physics_and_approach.md).




## Project idea

The workflow is intentionally compact and research-oriented:

- a dense neural network is trained to approximate the target-state fidelity,
- the trained surrogate is then used as a fast optimizer for the control parameters
- the optimal parameters found via the surrogate model are evaluated using a benchmark model that computes the exact dynamics


## Key findings

The current experiments show that the chosen architecture learns the fidelity landscape induced by the exact dynamics very well. The stored validation metrics indicate an average exact-surrogate difference of about $0.0066$.

In the present training pipeline, samples above a fidelity threshold of $0.7$ are excluded so that the model focuses on the harder low- and medium-fidelity region. Despite that restriction, the stored example optimization results still reach exact fidelities of about $0.94$, which is encouraging for more complex pulse sequences.

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
- loads the compact fidelity dataset,
- randomly samples up to $200000$ training examples,
- removes the fixed phase entries to obtain the five effective NN inputs,
- trains a dense neural network to approximate the physical benchmark,
- validates the predictions against the exact model and saves the trained surrogate.

### [Surrogate_Model_GD.py](Surrogate_Model_GD.py)
This script performs optimization using the trained surrogate model.

It:
- loads the trained network,
- computes gradients of the model output with respect to the input parameters,
- performs multiple Adam-based optimization runs from random initial points,
- validates the optimized parameters against the exact dynamical model,
- visualizes the comparison between neural-network predictions and true physical values.

## Data and artifacts

- [data/Three_Pulse_Dynamics_Data.npy](data/Three_Pulse_Dynamics_Data.npy): full benchmark dataset containing pulse parameters and state-vector output
- [data/Three_Pulse_Fidelity_Data.npy](data/Three_Pulse_Fidelity_Data.npy): compact dataset containing pulse parameters and one precomputed fidelity per sample
- [data/random_init_params.npy](data/random_init_params.npy): random initializations for optimization runs
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
