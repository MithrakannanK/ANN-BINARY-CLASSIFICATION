"""
Data loading and preprocessing module for the make_moons binary classification dataset.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


def generate_moon_dataset(
    n_samples: int = 1500,
    noise: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the two interleaving half circles (make_moons) synthetic dataset.

    Parameters:
    -----------
    n_samples : int
        The total number of points generated.
    noise : float
        Standard deviation of Gaussian noise added to the data.
    random_state : int
        Seed for reproducibility.

    Returns:
    --------
    X : np.ndarray of shape (n_samples, 2)
        Generated 2D feature coordinates.
    y : np.ndarray of shape (n_samples,)
        Integer class labels (0 or 1).
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def prepare_data_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
    scale_features: bool = True
) -> Dict[str, Any]:
    """
    Split data into Train, Validation, and Test partitions and scale features.

    Parameters:
    -----------
    X : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)
    test_size : float
        Fraction of data reserved for final testing.
    val_size : float
        Fraction of training set reserved for validation during training.
    random_state : int
        Reproducibility seed.
    scale_features : bool
        Whether to standardize features to zero mean and unit variance.

    Returns:
    --------
    data_dict : Dict containing train, val, and test arrays and fitted scaler.
    """
    # First split off the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Next split train into train and validation sets
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size,
        random_state=random_state, stratify=y_train_val
    )

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        X_test_scaled = X_test

    return {
        "X_raw": X,
        "y_raw": y,
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "X_train_orig": X_train,
        "X_val_orig": X_val,
        "X_test_orig": X_test,
        "scaler": scaler
    }


def create_pytorch_dataloaders(
    data_dict: Dict[str, Any],
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convert NumPy arrays to PyTorch TensorDatasets and DataLoaders.
    """
    # Convert numpy to float32 tensors
    train_x = torch.tensor(data_dict["X_train"], dtype=torch.float32)
    train_y = torch.tensor(data_dict["y_train"], dtype=torch.float32).unsqueeze(1)

    val_x = torch.tensor(data_dict["X_val"], dtype=torch.float32)
    val_y = torch.tensor(data_dict["y_val"], dtype=torch.float32).unsqueeze(1)

    test_x = torch.tensor(data_dict["X_test"], dtype=torch.float32)
    test_y = torch.tensor(data_dict["y_test"], dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    test_ds = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
