"""Quantum dynamics benchmark based on unitary time evolution.

This module constructs the operator matrices for the benchmark scheme,
diagonalizes the relevant Hamiltonians, and evaluates the probability of
preparing a desired Fock-state sector after a three-pulse sequence.
"""

import numpy as np


# -----------------------------------------------------------------------------
# Benchmark setup
# -----------------------------------------------------------------------------
dim = 200 # Hilbert space cutoff, i.e. maxmimum photon number in both modes
Omega = 1 # Rabi frequency of the JC interaction, sets the timescale of the dynamics


def ao_dag(N):
    """Construct the matrix representation of the atom-field interaction operator."""
    counter = 0
    array = np.zeros(2 * N - 1)

    for i in range(1, 2 * N - 1, 2):
        array[i] = np.sqrt(counter + 1)
        counter += 1

    return np.diag(array, +1)


def downdown(N):
    """Construct the matrix representation of the two-photon lowering term."""
    array = []

    for i in range(1, N):
        array.append(i)
        array.append(np.sqrt(i + 1) * np.sqrt(i))

    return np.diag(array, +2)


# Basic operator building blocks used in the Hamiltonians.
rabi_up = ao_dag(dim)
rabi_down = rabi_up.T  # Real-valued matrix, so transpose is sufficient.
down_down = downdown(dim)
up_up = down_down.T


def H_NL(phi):
    """Return the nonlinear Hamiltonian for a given phase phi."""
    H_NL = np.exp(1j * phi) * up_up + np.exp(-1j * phi) * down_down
    return H_NL


def H_2LS(Omega):
    """Return the effective two-level-system Hamiltonian."""
    H_2LS = Omega * 1j * (rabi_up - rabi_down) / 2
    return H_2LS


def Three_Pulse_Dynamics(params, ev_JC, V_JC, ev_NL1, V_NL1, ev_NL2, V_NL2):
    """Assemble the total unitary for the three-pulse benchmark sequence.

    Parameters are interpreted as:
    r1, phi1, t1, r2, phi2, t2, r3, phi3

    In the current benchmark setup, the nonlinear phases used in the actual
    Hamiltonians are fixed externally to 0 and pi. The full parameter vector is
    still unpacked here for clarity and compatibility with the broader workflow.
    """
    r1 = params[0]
    phi1 = params[1]
    t1 = params[2]
    r2 = params[3]
    phi2 = params[4]
    t2 = params[5]
    r3 = params[6]
    phi3 = params[7]

    # Prevent linter noise while keeping the full parameter semantics visible.
    _ = phi1, phi2, phi3

    # Define unitary matrices through diagonalization of the Hamiltonians.
    # First pulse
    U_NL1 = V_NL1 @ np.diag(np.exp(-1j * ev_NL1 * r1)) @ np.conj(V_NL1.T)
    # First Rabi period
    U_2LS1 = V_JC @ np.diag(np.exp(-1j * ev_JC * t1)) @ np.conj(V_JC.T)
    # Second pulse
    U_NL2 = V_NL2 @ np.diag(np.exp(-1j * ev_NL2 * r2)) @ np.conj(V_NL2.T)
    # Second Rabi period
    U_2LS2 = V_JC @ np.diag(np.exp(-1j * ev_JC * t2)) @ np.conj(V_JC.T)
    # Third pulse
    U_NL3 = V_NL1 @ np.diag(np.exp(-1j * ev_NL1 * r3)) @ np.conj(V_NL1.T)
    # H_NL3 = H_NL1 for the fixed phase choice 0, pi, 0.

    # Total dynamics of the full pulse sequence.
    U_tot = U_NL3 @ U_2LS2 @ U_NL2 @ U_2LS1 @ U_NL1

    return U_tot


# -----------------------------------------------------------------------------
# Precompute eigendecompositions used repeatedly during evaluation
# -----------------------------------------------------------------------------
H_JC = H_2LS(Omega)
ev_JC, V_JC = np.linalg.eigh(H_JC)

phi1, phi2 = 0, np.pi
H_NL1 = H_NL(phi1)
ev_NL1, V_NL1 = np.linalg.eigh(H_NL1)
H_NL2 = H_NL(phi2)
ev_NL2, V_NL2 = np.linalg.eigh(H_NL2)


def compute_P(N, params):
    """Compute the target-state probability after the three-pulse evolution."""
    U_tot = Three_Pulse_Dynamics(params, ev_JC, V_JC, ev_NL1, V_NL1, ev_NL2, V_NL2)
    vac = np.eye(2 * dim)[0]
    psi_f = U_tot @ vac
    P = np.sum((np.abs(psi_f) ** 2)[2 * N - 1 : 2 * N + 1])
    return P
