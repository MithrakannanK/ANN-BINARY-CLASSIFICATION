"""
PyTorch Implementation of an Artificial Neural Network for Moon Dataset Binary Classification.
"""

from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


class MoonClassifierPyTorch(nn.Module):
    """
    Multi-layer Perceptron (Artificial Neural Network) built with PyTorch.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: List[int] = [16, 8],
        output_dim: int = 1,
        activation: str = "relu",
        dropout_rate: float = 0.0
    ):
        super(MoonClassifierPyTorch, self).__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            elif activation.lower() == "gelu":
                layers.append(nn.GELU())
            elif activation.lower() == "leaky_relu":
                layers.append(nn.LeakyReLU(0.01))
            else:
                layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = h_dim

        # Final linear layer producing raw logits (for BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        return self.network(x)


class PyTorchANNWrapper:
    """
    Training and evaluation wrapper providing scikit-learn like fit/predict interface.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [16, 8],
        activation: str = "relu",
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        device: Optional[str] = None
    ):
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MoonClassifierPyTorch(
            input_dim=2,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation
        ).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 200,
        verbose: bool = True
    ) -> "PyTorchANNWrapper":
        """
        Train the model over epochs.
        """
        for epoch in range(1, epochs + 1):
            # Training Phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_b)
                loss = self.criterion(logits, y_b)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * X_b.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)

            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            self.history["train_loss"].append(epoch_train_loss)
            self.history["train_acc"].append(epoch_train_acc)

            # Validation Phase
            epoch_val_loss = None
            epoch_val_acc = None
            if val_loader is not None:
                self.model.eval()
                val_running_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                        val_logits = self.model(X_val)
                        v_loss = self.criterion(val_logits, y_val)

                        val_running_loss += v_loss.item() * X_val.size(0)
                        v_preds = (torch.sigmoid(val_logits) >= 0.5).float()
                        val_correct += (v_preds == y_val).sum().item()
                        val_total += y_val.size(0)

                epoch_val_loss = val_running_loss / val_total
                epoch_val_acc = val_correct / val_total
                self.history["val_loss"].append(epoch_val_loss)
                self.history["val_acc"].append(epoch_val_acc)

            if verbose and (epoch % 25 == 0 or epoch == 1 or epoch == epochs):
                msg = f"Epoch {epoch:3d}/{epochs} | PyTorch Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%"
                if epoch_val_loss is not None:
                    msg += f" | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%"
                print(msg)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of class 1 given 2D numpy array.
        """
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(tensor_x)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels (0 or 1)."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
