"""
Visualization module for dataset inspection, decision boundaries, training curves, and ROC curves.
"""

from typing import Dict, Any, List, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


# Use modern clean style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def plot_dataset_distribution(
    X: np.ndarray,
    y: np.ndarray,
    save_path: str = "plots/dataset_distribution.png"
) -> None:
    """
    Plot the 2D distribution of the make_moons dataset.
    """
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Class 0: Blue circles, Class 1: Orange triangles
    mask_0 = (y == 0)
    mask_1 = (y == 1)

    plt.scatter(
        X[mask_0, 0], X[mask_0, 1],
        color="#1f77b4", label="Class 0 (Upper Crescent)",
        alpha=0.75, edgecolors="k", s=50, linewidths=0.5
    )
    plt.scatter(
        X[mask_1, 0], X[mask_1, 1],
        color="#ff7f0e", label="Class 1 (Lower Crescent)",
        alpha=0.75, edgecolors="k", s=50, linewidths=0.5, marker="^"
    )

    plt.title("Scikit-learn 'make_moons' Binary Classification Dataset", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Feature 1 ($X_1$)", fontsize=12)
    plt.ylabel("Feature 2 ($X_2$)", fontsize=12)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved dataset distribution plot to: {save_path}")


def plot_decision_boundaries(
    models_dict: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    save_path: str = "plots/decision_boundaries.png"
) -> None:
    """
    Plot decision boundaries for multiple models side by side on a 2D meshgrid.
    """
    n_models = len(models_dict)
    cols = min(2, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), dpi=300)
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Create grid spanning the feature space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    for idx, (name, model) in enumerate(models_dict.items()):
        ax = axes[idx]

        # Get probabilities or predictions
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(grid_points)
            if probs.ndim == 2:
                probs = probs[:, 1]
            Z = probs.reshape(xx.shape)
        else:
            Z = model.predict(grid_points).reshape(xx.shape)

        # Plot contour decision surface
        contour = ax.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2.0, linestyles="--")

        # Scatter test points
        ax.scatter(
            X[y == 0, 0], X[y == 0, 1],
            color="#08519c", label="Class 0",
            alpha=0.8, edgecolors="white", s=40, linewidths=0.5
        )
        ax.scatter(
            X[y == 1, 0], X[y == 1, 1],
            color="#bd0026", label="Class 1",
            alpha=0.8, edgecolors="white", s=40, linewidths=0.5, marker="^"
        )

        ax.set_title(f"Decision Boundary: {name}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Feature 1 ($X_1$)")
        ax.set_ylabel("Feature 2 ($X_2$)")
        ax.legend(loc="upper right", frameon=True)

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.70])
    fig.colorbar(contour, cax=cbar_ax, label="Predicted Probability P(Class = 1)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved decision boundaries plot to: {save_path}")


def plot_training_curves(
    numpy_history: Dict[str, List[float]],
    pytorch_history: Optional[Dict[str, List[float]]] = None,
    save_path: str = "plots/training_curves.png"
) -> None:
    """
    Plot loss and accuracy learning curves across epochs.
    """
    n_plots = 2 if pytorch_history is not None else 1
    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 5 * n_plots), dpi=300)

    if n_plots == 1:
        axes = np.expand_dims(axes, axis=0)

    # Row 0: NumPy Scratch ANN
    epochs_np = range(1, len(numpy_history["train_loss"]) + 1)
    axes[0, 0].plot(epochs_np, numpy_history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    if "val_loss" in numpy_history and len(numpy_history["val_loss"]) > 0:
        axes[0, 0].plot(epochs_np, numpy_history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    axes[0, 0].set_title("NumPy ANN (Scratch): Loss vs Epochs", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs_np, [a * 100 for a in numpy_history["train_acc"]], label="Train Acc", color="#2ca02c", lw=2)
    if "val_acc" in numpy_history and len(numpy_history["val_acc"]) > 0:
        axes[0, 1].plot(epochs_np, [a * 100 for a in numpy_history["val_acc"]], label="Val Acc", color="#d62728", lw=2, linestyle="--")
    axes[0, 1].set_title("NumPy ANN (Scratch): Accuracy vs Epochs", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()

    # Row 1: PyTorch ANN (if present)
    if pytorch_history is not None:
        epochs_pt = range(1, len(pytorch_history["train_loss"]) + 1)
        axes[1, 0].plot(epochs_pt, pytorch_history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
        if "val_loss" in pytorch_history and len(pytorch_history["val_loss"]) > 0:
            axes[1, 0].plot(epochs_pt, pytorch_history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
        axes[1, 0].set_title("PyTorch ANN: Loss vs Epochs", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("BCE Loss")
        axes[1, 0].legend()

        axes[1, 1].plot(epochs_pt, [a * 100 for a in pytorch_history["train_acc"]], label="Train Acc", color="#2ca02c", lw=2)
        if "val_acc" in pytorch_history and len(pytorch_history["val_acc"]) > 0:
            axes[1, 1].plot(epochs_pt, [a * 100 for a in pytorch_history["val_acc"]], label="Val Acc", color="#d62728", lw=2, linestyle="--")
        axes[1, 1].set_title("PyTorch ANN: Accuracy vs Epochs", fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy (%)")
        axes[1, 1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to: {save_path}")


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: str = "plots/roc_curves.png"
) -> None:
    """
    Plot ROC Curves for multiple models on a single graph.
    roc_data: dict of model_name -> (fpr, tpr, roc_auc)
    """
    plt.figure(figsize=(8, 6), dpi=300)

    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
    for i, (name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
        color = colors[i % len(colors)]
        style = "--" if "Logistic" in name else "-"
        plt.plot(fpr, tpr, color=color, lw=2.2, linestyle=style, label=f"{name} (AUC = {roc_auc:.4f})")

    # Diagonal chance line
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle=":", label="Random Guess (AUC = 0.5000)")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, facecolor="white", shadow=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves to: {save_path}")


def plot_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    save_path: str = "plots/confusion_matrices.png"
) -> None:
    """
    Plot confusion matrices for all evaluated models.
    """
    n_models = len(confusion_matrices)
    cols = min(2, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), dpi=300)
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticklabels(["Class 0", "Class 1"])
        ax.set_yticklabels(["Class 0", "Class 1"])

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices to: {save_path}")
