# Physics background and surrogate-model approach

This document provides a more detailed description of the physical model and the machine-learning workflow used in this repository.

## Hybrid quantum system

We consider a hybrid quantum system consisting of a two-level system and a nonlinear crystal inside an optical cavity. The cavity supports two modes, namely a signal mode and an idler mode. The idler mode is resonant with the transition frequency of the two-level system, while the signal mode is detuned.
Properly engineered pump pulses enable the preparation of high-fidelity Fock states in the signal mode.

The dynamics are governed by two interaction mechanisms:

- atom-field coupling in the idler mode, which induces Rabi oscillations through the Jaynes-Cummings interaction,
- pump-driven nonlinear interaction in the crystal, which generates two-mode squeezing of the signal and idler modes.

## Effective evolution for three pulses

The effective unitary dynamics are approximated by

$$
\hat{U} \approx
\hat{S}(r_3,\phi_3)\hat{T}(t_2)\hat{S}(r_2,\phi_2)\hat{T}(t_1)\hat{S}(r_1,\phi_1)\.
$$

Here, the parameters $r_j$, $\phi_j$, and $t_j$ denote the pulse gains, phases, and intermediate interaction times.<br>
The operators $\hat{S}, \hat{T}$ describe the two interactions mechanisms.

## Two-mode squeezing operator

The nonlinear interaction modeling a single pump pulse is described by the two-mode squeezing operator

$$
\hat{S}(r_j,\phi_j)=\exp\left[-i r_j \left(e^{i\phi_j}\hat{a}^\dagger_i \hat{a}^\dagger_s + e^{-i\phi_j}\hat{a}_i \hat{a}_s \right)\right]\.
$$


This operator depends on the pump pulse parameters $r_j, \phi_j$ and the symbols $\hat{a}_i$ and $\hat{a}_s$ denote the annihilation operators of the idler and signal modes.

## Atom-field interaction

Between pulses, the atom-field interaction is modeled by

$$
\hat{T}(t_j) = \exp\left[t_j \frac{\Omega}{2}(\hat{a}_i \hat{\sigma}^\dagger - \hat{a}_i^\dagger \hat{\sigma})\right]\,
$$

and depends on the time-delay parameter $t_j$. $\Omega$ is the single-photon Rabi frequency and $\hat{\sigma} = |g\rangle\langle e|$ is the atomic lowering operator.


## Fidelity definition

The figure of merit is the fidelity with respect to a target Fock state $|N\rangle$ in the signal mode:

$$
F(N) = \mathrm{Tr}\left[
|N\rangle\langle N|
\;\mathrm{Tr}_{i,\mathrm{atom}}(\rho_{\mathrm{final}})
\right],
$$

which is obtained by taking the partial trace of the final state density operator over the idler and atomic degrees of freedom. The final state density operator is given by

$$
\rho_{\mathrm{final}} = \hat{U}|0,0,g\rangle\langle 0,0,g|\hat{U}^\dagger.
$$

## Surrogate-model approach

This project introduces a surrogate model based on a dense feed-forward neural network. The network is trained in a supervised-learning setting to approximate the mapping

$$
f^{(N)}_{\mathrm{NN}}(\vec{\theta}; \omega) \approx F(N;\vec{\theta}),
$$

where $\vec{\theta}$ denotes the control parameters and $\omega$ the trainable network weights.

After training, the neural network acts as a fast differentiable proxy for the expensive benchmark simulation. This makes it possible to optimize the pulse parameters directly through the surrogate by minimizing

$$
\mathcal{L} = 1 - f^{(N)}_{\mathrm{NN}}(\vec{\theta}; \omega).
$$

In the implementation, the pulse parameters are updated with the Adam optimizer using the surrogate gradients.

## Current project scope and findings

The present repository focuses on a deliberately simple but informative setting:

- target Fock state $N=1$,
- a three-pulse protocol,
- fixed phases of the pulses, which reduces the surrogate input to five effective parameters.

This reduced setting is still challenging enough to test whether the neural network learns the fidelity landscape induced by the full dynamics.

The current results show that the chosen architecture learns the benchmark fidelities very accurately. According to the stored training metrics, the average validation difference is about $0.0066$. In addition, the training pipeline explicitly filters out samples above a fidelity threshold of $0.7$, so the surrogate is trained mainly on the more difficult low- to medium-fidelity region.

Even under that restriction, surrogate-guided gradient optimization can still discover parameter sets with high exact fidelities. In the current experiments, the best exact fidelity found after optimization is approximately $0.94$.

## Outlook

These findings suggest that surrogate-based optimization is promising well beyond the current proof-of-concept setting. If the same idea remains robust for larger parameter spaces, it could become especially useful for more complicated pulse sequences, higher target Fock states, or models that include additional physical effects such as dissipation and loss.
