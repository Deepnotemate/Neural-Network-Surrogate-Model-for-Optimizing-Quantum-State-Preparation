"""Gradient-based optimization of control parameters using the surrogate model.

This script performs gradient ascent on the trained neural-network surrogate,
starting from multiple random initializations. The optimized parameters are then
validated against the exact physical benchmark.
"""

import json
import os
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from Dynamics_with_unitary_operators import compute_P


# -----------------------------------------------------------------------------
# Load trained surrogate model
# -----------------------------------------------------------------------------
data_dir = Path("data")
artifacts_dir = Path("artifacts")
figures_dir = artifacts_dir / "figures"
metrics_dir = artifacts_dir / "metrics"
models_dir = artifacts_dir / "models"

for directory in (figures_dir, metrics_dir, models_dir):
    directory.mkdir(parents=True, exist_ok=True)

model_path = models_dir / "surrogate_model.h5"
if not model_path.exists():
    model_path = Path("surrogate_model.h5")

model = keras.models.load_model(model_path)

TARGET_FOCK_STATE = 1  # Specify the target Fock state here.


# -----------------------------------------------------------------------------
# Gradient-based optimization utilities
# -----------------------------------------------------------------------------


def Adam(
    iteration_number,
    params,
    beta1,
    beta2,
    s,
    m,
    learning_rate,
    gradient,
    epsilon=10**-7,
):
    """Perform one Adam update step on the input parameter vector."""
    iteration_number = tf.cast(iteration_number, tf.float32)
    params = tf.cast(params, tf.float32)
    gradient = tf.cast(gradient, tf.float32)
    s = tf.cast(s, tf.float32)
    m = tf.cast(m, tf.float32)

    m = beta1 * m - (1 - beta1) * (-gradient)
    s = beta2 * s + (1 - beta2) * tf.square(-gradient)
    m_hat = m / (1 - beta1**iteration_number)
    s_hat = s / (1 - beta2**iteration_number)
    out = params + learning_rate * m_hat / tf.sqrt(s_hat + epsilon)
    return m, s, out


@tf.function(reduce_retracing=True)
def compute_gradient(params):
    """Compute the surrogate-model gradient with respect to the input parameters."""
    with tf.GradientTape() as tape:
        tape.watch(params)
        y_pred = model(params, training=False)

    dy_dx = tape.gradient(y_pred, params)
    return dy_dx


def optimize_all_runs(num_iterations, init_params, learning_rate):
    """Optimize all random initializations simultaneously as one TensorFlow batch."""
    params = tf.Variable(init_params, dtype=tf.float32)
    s = tf.zeros_like(params)
    m = tf.zeros_like(params)

    for iteration in tqdm(range(num_iterations), desc="GD steps"):
        grad = compute_gradient(params)
        m, s, updated_params = Adam(iteration + 1, params, beta1, beta2, s, m, learning_rate, grad)
        params.assign(updated_params)

    return params.numpy()


def single_run(num_iterations, init_params, learning_rate):
    """Run one optimization trajectory from a single initialization."""
    optimized = optimize_all_runs(num_iterations, tf.convert_to_tensor(init_params, dtype=tf.float32), learning_rate)
    return optimized




# -----------------------------------------------------------------------------
# Optimization settings
# -----------------------------------------------------------------------------
learning_rate = 0.005
beta1 = 0.9
beta2 = 0.999
num_iterations = 500
num_runs = 50






def expand_to_full_params(params):
    """Map the reduced 5-parameter representation back to the full pulse vector."""
    params_h = np.insert(params, [1, 3], [0, np.pi])
    params_h = np.append(params_h, 0)
    return params_h


def evaluate_exact_fidelities(reduced_parameter_sets, target_fock_state):
    """Evaluate exact benchmark fidelities for a batch of reduced parameter vectors."""
    exact_values = []
    for params in reduced_parameter_sets:
        params_h = expand_to_full_params(params)
        exact_values.append(compute_P(target_fock_state, params_h))
    return np.array(exact_values)


def plot_gd_comparison(initial_exact_fidelities, optimized_exact_fidelities, surrogate_predictions):
    """Create a compact summary plot for surrogate-guided optimization results."""
    _ = initial_exact_fidelities
    mean_abs_difference = float(np.mean(np.abs(surrogate_predictions - optimized_exact_fidelities)))
    sorted_indices = np.argsort(surrogate_predictions)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    # Optimized predictions versus exact fidelities.
    x_positions = np.arange(1, len(surrogate_predictions) + 1)
    ax[0].bar(
        x_positions,
        optimized_exact_fidelities[sorted_indices],
        alpha=0.75,
        label="Exact",
    )
    ax[0].plot(
        x_positions,
        surrogate_predictions[sorted_indices],
        "o",
        color="black",
        markersize=4,
        label="Surrogate",
    )
    ax[0].set_title("Found fidelities")
    ax[0].set_xlabel("Optimization run")
    ax[0].set_ylabel("Fidelity")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    # Exact versus surrogate scatter plot.
    min_val = min(np.min(optimized_exact_fidelities), np.min(surrogate_predictions))
    max_val = max(np.max(optimized_exact_fidelities), np.max(surrogate_predictions))
    ax[1].scatter(optimized_exact_fidelities, surrogate_predictions, alpha=0.75, s=35)
    ax[1].plot([min_val, max_val], [min_val, max_val], "--", color="black", linewidth=1.2)
    ax[1].set_title("Exact vs surrogate inference")
    ax[1].set_xlabel("Exact fidelity from dynamics")
    ax[1].set_ylabel("Surrogate prediction")
    ax[1].grid(alpha=0.3)
    ax[1].text(
        0.04,
        0.96,
        f"Average difference = {mean_abs_difference:.4f}",
        transform=ax[1].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )

    fig.tight_layout()
    fig.savefig(figures_dir / "gd_fidelity_comparison.png", bbox_inches="tight")
    return fig, mean_abs_difference


def main():
    """Optimize surrogate parameters and compare them to the exact dynamics."""
    init_params = np.load(data_dir / "random_init_params.npy")[:num_runs]

    initial_exact_fidelities = evaluate_exact_fidelities(init_params, TARGET_FOCK_STATE)

    start_time = time.perf_counter()
    optimized_params = optimize_all_runs(num_iterations, init_params, learning_rate)
    optimization_time_seconds = time.perf_counter() - start_time

    surrogate_predictions = model.predict(optimized_params, verbose=0)[:, 0]
    optimized_exact_fidelities = evaluate_exact_fidelities(optimized_params, TARGET_FOCK_STATE)

    _, mean_abs_difference = plot_gd_comparison(
        initial_exact_fidelities,
        optimized_exact_fidelities,
        surrogate_predictions,
    )

    summary_payload = {
        "target_fock_state": int(TARGET_FOCK_STATE),
        "num_runs": int(num_runs),
        "num_iterations": int(num_iterations),
        "learning_rate": float(learning_rate),
        "best_initial_exact_fidelity": float(np.max(initial_exact_fidelities)),
        "best_optimized_exact_fidelity": float(np.max(optimized_exact_fidelities)),
        "mean_initial_exact_fidelity": float(np.mean(initial_exact_fidelities)),
        "mean_optimized_exact_fidelity": float(np.mean(optimized_exact_fidelities)),
        "best_surrogate_prediction": float(np.max(surrogate_predictions)),
        "average_exact_surrogate_difference": mean_abs_difference,
        "optimization_time_seconds": float(optimization_time_seconds),
    }

    with open(metrics_dir / "gd_optimization_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2)

    np.save(metrics_dir / "gd_initial_exact_fidelities.npy", initial_exact_fidelities)
    np.save(metrics_dir / "gd_optimized_exact_fidelities.npy", optimized_exact_fidelities)
    np.save(metrics_dir / "gd_surrogate_predictions.npy", surrogate_predictions)
    np.save(metrics_dir / "gd_optimized_params.npy", optimized_params)

    print(
        f"Best exact fidelity after GD: {np.max(optimized_exact_fidelities):.4f} | "
    )
    print(f"Best surrogate prediction: {np.max(surrogate_predictions):.4f}")
    print(f"Average exact-surrogate difference: {mean_abs_difference:.4f}")
    print(f"Optimization time: {optimization_time_seconds:.2f} s")
    print(f"Saved optimization figure to: {figures_dir / 'gd_fidelity_comparison.png'}")

    plt.show()


if __name__ == "__main__":
    main()

