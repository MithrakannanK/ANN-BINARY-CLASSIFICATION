"""
Master execution pipeline for Artificial Neural Network Binary Classification on make_moons.
Trains and compares:
1. Baseline Linear Classifier (Logistic Regression)
2. NumPy ANN from scratch (Backpropagation & Adam)
3. PyTorch Deep Learning ANN
4. Scikit-Learn Tuned MLPClassifier
Generates and saves all evaluation plots and metrics.
"""

import sys
import os
import numpy as np
from sklearn.metrics import roc_curve, auc

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    generate_moon_dataset,
    prepare_data_splits,
    create_pytorch_dataloaders
)
from src.ann_numpy import NeuralNetworkFromScratch
from src.ann_pytorch import PyTorchANNWrapper
from src.ann_sklearn import (
    train_baseline_linear_model,
    train_sklearn_mlp,
    tune_mlp_hyperparameters
)
from src.evaluation import (
    compute_classification_metrics,
    format_metrics_table
)
from src.visualization import (
    plot_dataset_distribution,
    plot_decision_boundaries,
    plot_training_curves,
    plot_roc_curves,
    plot_confusion_matrices
)


def main():
    print("=" * 75)
    print(" ARTIFICIAL NEURAL NETWORK (ANN) BINARY CLASSIFICATION PIPELINE ")
    print("             Dataset: Scikit-learn 'make_moons'                 ")
    print("=" * 75)

    # 1. Generate Dataset
    print("\n[Step 1/6] Generating non-linear 'make_moons' dataset...")
    n_samples = 1500
    noise = 0.20
    random_state = 42

    X_raw, y_raw = generate_moon_dataset(n_samples=n_samples, noise=noise, random_state=random_state)
    print(f"Generated {X_raw.shape[0]} samples with 2 input features and noise std={noise}.")

    # Visualize and save raw dataset
    plot_dataset_distribution(X_raw, y_raw, save_path="plots/dataset_distribution.png")

    # 2. Partition & Preprocess Data
    print("\n[Step 2/6] Partitioning data into Train, Validation, and Test sets & Standardizing features...")
    data = prepare_data_splits(
        X_raw, y_raw, test_size=0.20, val_size=0.15, random_state=random_state, scale_features=True
    )

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"Training set size   : {X_train.shape[0]} samples")
    print(f"Validation set size : {X_val.shape[0]} samples")
    print(f"Test set size       : {X_test.shape[0]} samples")

    # 3. Model Training
    models = {}
    metrics_list = []
    roc_dict = {}
    cm_dict = {}

    # --- Model A: Baseline Logistic Regression ---
    print("\n--- Training Model A: Baseline Linear Classifier (Logistic Regression) ---")
    baseline_lr = train_baseline_linear_model(X_train, y_train)
    models["Baseline (Logistic Regression)"] = baseline_lr

    # --- Model B: NumPy ANN from Scratch ---
    print("\n--- Training Model B: NumPy ANN From Scratch ([2, 16, 8, 1], ReLU, Adam) ---")
    ann_scratch = NeuralNetworkFromScratch(
        layer_sizes=[2, 16, 8, 1],
        hidden_activation="relu",
        l2_reg=0.001,
        optimizer="adam",
        learning_rate=0.02,
        seed=random_state
    )
    ann_scratch.fit(
        X=X_train,
        y=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=350,
        batch_size=32,
        verbose=True
    )
    models["NumPy Scratch ANN"] = ann_scratch

    # --- Model C: PyTorch ANN ---
    print("\n--- Training Model C: PyTorch Deep Learning ANN ([2, 16, 8, 1], AdamW) ---")
    train_loader, val_loader, test_loader = create_pytorch_dataloaders(data, batch_size=32)
    pytorch_ann = PyTorchANNWrapper(
        hidden_dims=[16, 8],
        activation="relu",
        lr=0.015,
        weight_decay=1e-4
    )
    pytorch_ann.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=150,
        verbose=True
    )
    models["PyTorch ANN"] = pytorch_ann

    # --- Model D: Scikit-Learn Tuned MLPClassifier ---
    print("\n--- Training Model D: Scikit-Learn MLPClassifier with GridSearchCV ---")
    grid_search = tune_mlp_hyperparameters(X_train, y_train, cv_folds=5)
    best_mlp = grid_search.best_estimator_
    print(f"Best MLP Parameters found: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_*100:.2f}%")
    models["Scikit-Learn Tuned MLP"] = best_mlp

    # 4. Evaluation on Unseen Test Set
    print("\n[Step 4/6] Evaluating all models on the unseen Test partition...")
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            if probs.ndim == 2:
                probs = probs[:, 1]
            preds = (probs >= 0.5).astype(int)
        else:
            preds = model.predict(X_test)
            probs = preds

        metrics = compute_classification_metrics(y_test, preds, probs, model_name=name)
        metrics_list.append(metrics)

        # ROC Curve data
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_dict[name] = (fpr, tpr, metrics["roc_auc"])

        # Confusion Matrix
        cm_dict[name] = metrics["confusion_matrix"]

    # 5. Display Summary Table
    print("\n" + "=" * 80)
    print("                     TEST SET PERFORMANCE COMPARISON                     ")
    print("=" * 80)
    print(format_metrics_table(metrics_list))
    print("=" * 80)

    # 6. Generate and Save Visualizations
    print("\n[Step 6/6] Generating publication-quality visualization plots...")
    plot_decision_boundaries(models, X_test, y_test, save_path="plots/decision_boundaries.png")
    plot_training_curves(ann_scratch.history, pytorch_ann.history, save_path="plots/training_curves.png")
    plot_roc_curves(roc_dict, save_path="plots/roc_curves.png")
    plot_confusion_matrices(cm_dict, save_path="plots/confusion_matrices.png")

    print("\nAll tasks completed successfully! Generated plots in 'plots/' directory.")


if __name__ == "__main__":
    main()
