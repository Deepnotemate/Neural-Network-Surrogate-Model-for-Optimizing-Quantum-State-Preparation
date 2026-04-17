"""Lightweight tests for the exact quantum-dynamics benchmark.

These tests focus on mathematical consistency and a small set of regression
values. They are not meant to be exhaustive, but should catch major issues with the dynamics code, such as non-unitary evolution or incorrect operator definitions.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Dynamics_with_unitary_operators as dyn


EXAMPLE_REFERENCE_CASES = [
    (
        1,
        np.array([0.10, 0.0, 0.20, 0.15, np.pi, 0.25, 0.10, 0.0], dtype=float),
        0.0023699265761740066,
    ),
    (
        1,
        np.array([0.40, 0.0, 0.35, 0.20, np.pi, 0.10, 0.25, 0.0], dtype=float),
        0.14507446224273365,
    ),
]

HIGH_FIDELITY_EXAMPLE_CASES = [
    (
        "case_1",
        1,
        np.array([0.54801525, 6.97433569, 1.48056221, 1.19380521, 1.42645147], dtype=float),
        98,
    ),
    (
        "case_2",
        2,
        np.array([0.93254696, 8.85929128, 1.48056221, 2.136283, 1.26181663], dtype=float),
        93,
    ),
    (
        "case_3",
        3,
        np.array([1.0971818, 9.36194611, 1.53582426, 3.14159265, 1.15129255], dtype=float),
        85,
    ),
    (
        "case_4",
        4,
        np.array([1.0971818, 9.36194611, 1.53582426, 4.08407045, 1.0971818], dtype=float),
        74,
    ),
]


def expand_reduced_params(reduced_params):
    """Expand the 5-parameter representation to the full 8-parameter vector that includes phase parameters."""
    full_params = np.insert(reduced_params, [1, 3], [0, np.pi])
    full_params = np.append(full_params, 0)
    return full_params

def test_hamiltonians_are_hermitian():
    """The effective Hamiltonians should be Hermitian."""
    h_nl = dyn.H_NL(np.pi / 3)
    h_2ls = dyn.H_2LS(dyn.Omega)

    assert np.allclose(h_nl, h_nl.conj().T)
    assert np.allclose(h_2ls, h_2ls.conj().T)


def test_three_pulse_dynamics_is_unitary():
    """The composed time-evolution operator should remain unitary."""
    params = np.array([0.15, 0.0, 0.20, 0.10, np.pi, 0.30, 0.12, 0.0], dtype=float)
    u_tot = dyn.Three_Pulse_Dynamics(
        params,
        dyn.ev_JC,
        dyn.V_JC,
        dyn.ev_NL1,
        dyn.V_NL1,
        dyn.ev_NL2,
        dyn.V_NL2,
    )

    identity = np.eye(u_tot.shape[0], dtype=complex)
    assert np.allclose(u_tot.conj().T @ u_tot, identity, atol=1e-8)


@pytest.mark.parametrize("target_n, params, expected_probability", EXAMPLE_REFERENCE_CASES)
def test_reference_probabilities(target_n, params, expected_probability):
    """Known parameter sets should keep returning stable fidelities."""
    probability = dyn.compute_P(target_n, params)
    assert np.isclose(probability, expected_probability, atol=1e-12)


def test_probability_is_bounded_between_zero_and_one():
    """Physical fidelities must stay in the interval [0, 1]."""
    params = np.array([0.25, 0.0, 0.15, 0.18, np.pi, 0.10, 0.22, 0.0], dtype=float)
    probability = dyn.compute_P(1, params)
    assert 0.0 <= probability <= 1.0 + 1e-12


@pytest.mark.parametrize(
    "case_name, target_n, reduced_params, reported_fidelity_percent",
    HIGH_FIDELITY_EXAMPLE_CASES,
)
def test_user_provided_cases_match_reported_fidelities(
    case_name,
    target_n,
    reduced_params,
    reported_fidelity_percent,
):
    """User-supplied cases should reproduce the reported target-state fidelities."""
    _ = case_name
    full_params = expand_reduced_params(reduced_params)
    probability = dyn.compute_P(target_n, full_params)
    assert np.isclose(probability * 100, reported_fidelity_percent, atol=1.0)




