import numpy as np
import matplotlib.pyplot as plt

from src.models import create_cnn
from src.training import compile_model, create_early_stopping


def train_candidate(
    X_fit,
    y_fit,
    X_validation,
    y_validation,
    *,
    learning_rate,
    dropout_rate,
    batch_size,
    epochs=15,
    patience=2,
    verbose=1,
):
    model = compile_model(
        create_cnn(dropout_rate=dropout_rate),
        learning_rate=learning_rate,
    )
    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[create_early_stopping(patience=patience)],
        verbose=verbose,
    )
    return {
        "model": model,
        "history": history,
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "val_accuracy": float(max(history.history["val_accuracy"])),
        "val_loss": float(min(history.history["val_loss"])),
    }


def select_by_accuracy_and_loss(results, accuracy_tolerance=0.0005):
    """Choose lowest validation loss among near-tied accuracy candidates.

    ``accuracy_tolerance`` is an absolute difference on the 0–1 accuracy scale;
    0.0005 corresponds to 0.05 percentage points.
    """
    best_accuracy = max(result["val_accuracy"] for result in results.values())
    candidates = [
        value for value, result in results.items()
        if best_accuracy - result["val_accuracy"] <= accuracy_tolerance
    ]
    selected = min(candidates, key=lambda value: results[value]["val_loss"])
    print(f"Best validation accuracy: {best_accuracy:.4%}")
    print(
        f"Candidates within {accuracy_tolerance * 100:.2f} percentage points: "
        f"{candidates}"
    )
    print(
        f"Selected {selected}: lowest validation loss "
        f"({results[selected]['val_loss']:.5f}) among near-tied candidates; "
        f"validation accuracy={results[selected]['val_accuracy']:.4%}."
    )
    return selected


def plot_comparison(results, title, xlabel, selected_value):
    """Compare validation accuracy and loss, outlining the selected candidate."""
    labels = [str(value) for value in results]
    accuracies = [results[value]["val_accuracy"] for value in results]
    losses = [results[value]["val_loss"] for value in results]
    selected_index = list(results).index(selected_value)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, values, metric, color in [
        (axes[0], accuracies, "Validation accuracy", "#4C78A8"),
        (axes[1], losses, "Validation loss", "#F58518"),
    ]:
        bars = ax.bar(labels, values, color=color, alpha=0.85)
        bars[selected_index].set_edgecolor("black")
        bars[selected_index].set_linewidth(2.5)
        ax.set_title(metric)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axes[0].set_ylabel("Higher is better")
    axes[1].set_ylabel("Lower is better")
    fig.suptitle(f"{title} — combined selection: {selected_value}")
    fig.tight_layout()
    plt.show()

def candidate_summary(results):
    return {
        str(value): {
            "validation_accuracy": result["val_accuracy"],
            "validation_loss": result["val_loss"],
            "best_validation_loss_epoch": result["best_epoch"],
        }
        for value, result in results.items()
    }
