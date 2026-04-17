"""Training script for the surrogate neural network model.

This script trains the surrogate on a sampled subset of the exact dynamics
dataset, stores the resulting model in a dedicated artifacts folder, and creates
plots that summarize training and validation performance.
"""

import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback

from Dynamics_with_unitary_operators import compute_P


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
num_params = 5  # 3 pulses
N = 1
num_samples = 200000
batch_size = 64
epochs = 3#180
validation_eval_samples = 250

data_dir = Path("data")
artifacts_dir = Path("artifacts")
models_dir = artifacts_dir / "models"
figures_dir = artifacts_dir / "figures"
metrics_dir = artifacts_dir / "metrics"

for directory in (models_dir, figures_dir, metrics_dir):
    directory.mkdir(parents=True, exist_ok=True)


def build_model():
    """Build and compile the surrogate network."""
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(num_params,)))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=["mean_squared_error"],
    )
    return model


def expand_to_full_params(reduced_params):
    """Map the 5-dimensional NN input back to the full 8-parameter pulse vector."""
    full_params = np.insert(reduced_params, [1, 3], [0, np.pi])
    full_params = np.append(full_params, 0)
    return full_params


def plot_training_history(history):
    """Plot training and validation loss across epochs."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(history.history["loss"], label="Training loss", linewidth=2)
    ax.plot(history.history["val_loss"], label="Validation loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared error")
    ax.set_title("Training history")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "training_history.png", bbox_inches="tight")
    return fig


def plot_validation_comparison(exact_fidelities, surrogate_predictions):
    """Plot exact validation fidelities against surrogate predictions."""
    mean_abs_difference = float(np.mean(np.abs(surrogate_predictions - exact_fidelities)))

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    ax.scatter(exact_fidelities, surrogate_predictions, alpha=0.7, s=35)

    min_val = min(np.min(exact_fidelities), np.min(surrogate_predictions))
    max_val = max(np.max(exact_fidelities), np.max(surrogate_predictions))
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="black", linewidth=1.5)

    ax.set_xlabel("Exact fidelity from dynamics")
    ax.set_ylabel("Surrogate prediction")
    ax.set_title("Validation-set inference")
    ax.grid(alpha=0.3)
    ax.text(
        0.04,
        0.96,
        f"Average difference = {mean_abs_difference:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )

    fig.tight_layout()
    fig.savefig(figures_dir / "validation_exact_vs_surrogate.png", bbox_inches="tight")
    return fig, mean_abs_difference


class BatchLossHistory(Callback):
    """Track batch-wise training loss during fitting."""

    def on_train_begin(self, logs=None):
        self.batch_losses = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get("loss"))


def main():
    """Train the surrogate model and store artifacts for the repository."""
    model = build_model()

    # Load the compact dataset: 8 pulse parameters + 1 precomputed fidelity value.
    loaded_data = np.load(data_dir / "Three_Pulse_Dynamics_Data.npy", mmap_mode="r")

    # Random subsampling
    rng = np.random.default_rng(42)
    sample_count = min(num_samples, loaded_data.shape[0])
    indices = rng.choice(loaded_data.shape[0], size=sample_count, replace=False)
    sampled_rows = loaded_data[indices]

    full_params = sampled_rows[:, :8].astype(np.float32)
    sampled_labels = sampled_rows[:, 8].astype(np.float32)
    sampled_data = np.delete(full_params, [1, 4, 7], axis=1)  # remove fixed phase inputs

    # Remove high-fidelity samples to focus training on the more challenging low-fidelity region.
    fidelity_threshold = 0.7
    low_fidelity_mask = sampled_labels <= fidelity_threshold

    sampled_labels = sampled_labels[low_fidelity_mask]
    sampled_data = sampled_data[low_fidelity_mask]



    # Train-test split
    Training_Data, Test_Data, Training_labels, Test_labels = train_test_split(
        sampled_data,
        sampled_labels,
        test_size=0.2,
        random_state=42,
    )

    batch_history = BatchLossHistory()

    # Model training
    history = model.fit(
        Training_Data,
        Training_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(Test_Data, Test_labels),
        verbose=1,
        callbacks=[batch_history],
    )

    # Save trained model artifacts
    model_path = models_dir / "surrogate_model.h5"
    model.save(model_path)
    model.save("surrogate_model.h5")  # compatibility copy for downstream scripts

    # Exact-vs-surrogate validation on a representative subset
    eval_count = min(validation_eval_samples, len(Test_Data))
    eval_indices = rng.choice(len(Test_Data), size=eval_count, replace=False)
    validation_inputs = Test_Data[eval_indices]
    surrogate_predictions = model.predict(validation_inputs, verbose=0)[:, 0]

    exact_fidelities = []
    for params in validation_inputs:
        full_params = expand_to_full_params(params)
        exact_fidelities.append(compute_P(N, full_params))
    exact_fidelities = np.array(exact_fidelities)

    _, mean_abs_difference = plot_validation_comparison(exact_fidelities, surrogate_predictions)
    plot_training_history(history)

    # Store summary data for later inspection or README figures
    history_payload = {
        "loss": [float(value) for value in history.history.get("loss", [])],
        "val_loss": [float(value) for value in history.history.get("val_loss", [])],
        "batch_loss": [float(value) for value in batch_history.batch_losses if value is not None],
    }

    summary_payload = {
        "target_fock_state": int(N),
        "num_training_samples": int(len(Training_Data)),
        "num_validation_samples": int(len(Test_Data)),
        "validation_eval_subset": int(eval_count),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "final_training_loss": float(history.history["loss"][-1]),
        "final_validation_loss": float(history.history["val_loss"][-1]),
        "average_validation_difference": mean_abs_difference,
    }

    with open(metrics_dir / "training_history.json", "w", encoding="utf-8") as file:
        json.dump(history_payload, file, indent=2)

    with open(metrics_dir / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2)

    np.save(metrics_dir / "validation_exact_fidelities.npy", exact_fidelities)
    np.save(metrics_dir / "validation_surrogate_predictions.npy", surrogate_predictions)

    print(f"Saved trained model to: {model_path}")
    print(f"Saved figures to: {figures_dir}")
    print(f"Average validation difference: {mean_abs_difference:.4f}")

    plt.show()


if __name__ == "__main__":
    main()