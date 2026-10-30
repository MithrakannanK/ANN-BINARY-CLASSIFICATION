"""
Artificial Neural Network (ANN) implemented from scratch using NumPy.
Includes modular activation functions, He/Xavier initialization, 
Binary Cross-Entropy Loss with L2 regularization, and modern optimizers (SGD, Momentum, Adam).
"""

from typing import List, Tuple, Dict, Optional, Literal
import numpy as np


class NeuralNetworkFromScratch:
    """
    Fully-connected Feedforward Artificial Neural Network for Binary Classification.
    """

    def __init__(
        self,
        layer_sizes: List[int] = [2, 16, 8, 1],
        hidden_activation: Literal["relu", "tanh", "leaky_relu"] = "relu",
        l2_reg: float = 0.001,
        optimizer: Literal["adam", "sgd", "momentum"] = "adam",
        learning_rate: float = 0.01,
        seed: int = 42
    ):
        """
        Parameters:
        -----------
        layer_sizes : List[int]
            Sizes of input layer, hidden layers, and output layer (e.g., [2, 16, 8, 1]).
        hidden_activation : str
            Activation function for hidden layers ('relu', 'tanh', or 'leaky_relu').
        l2_reg : float
            L2 regularization strength (lambda).
        optimizer : str
            Optimization algorithm ('adam', 'sgd', or 'momentum').
        learning_rate : float
            Initial learning rate for parameter updates.
        seed : int
            Random seed for reproducible weight initialization.
        """
        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation.lower()
        self.l2_reg = l2_reg
        self.optimizer = optimizer.lower()
        self.learning_rate = learning_rate
        self.seed = seed

        self.num_layers = len(layer_sizes) - 1  # Number of weight layers
        self.params: Dict[str, np.ndarray] = {}
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }

        # Optimizer state variables
        self.v: Dict[str, np.ndarray] = {}  # Velocity / 1st moment
        self.s: Dict[str, np.ndarray] = {}  # 2nd moment (Adam)
        self.t: int = 0                     # Timestep for Adam bias correction

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """
        Initialize weights using He (for ReLU/LeakyReLU) or Xavier (for Tanh),
        and initialize biases to zero.
        """
        np.random.seed(self.seed)

        for l in range(1, len(self.layer_sizes)):
            n_in = self.layer_sizes[l - 1]
            n_out = self.layer_sizes[l]

            if self.hidden_activation in ["relu", "leaky_relu"] and l < self.num_layers:
                # He (Kaiming) initialization: variance = 2 / n_in
                std = np.sqrt(2.0 / n_in)
            else:
                # Xavier / Glorot initialization: variance = 1 / n_in
                std = np.sqrt(1.0 / n_in)

            self.params[f"W{l}"] = np.random.randn(n_in, n_out) * std
            self.params[f"b{l}"] = np.zeros((1, n_out))

            # Optimizer states
            self.v[f"dW{l}"] = np.zeros_like(self.params[f"W{l}"])
            self.v[f"db{l}"] = np.zeros_like(self.params[f"b{l}"])
            self.s[f"dW{l}"] = np.zeros_like(self.params[f"W{l}"])
            self.s[f"db{l}"] = np.zeros_like(self.params[f"b{l}"])

    # ---------------- Activation Functions ---------------- #

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        # Clip z to avoid overflow in exp(-z)
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _apply_activation(self, z: np.ndarray, activation: str) -> np.ndarray:
        """Apply specified non-linear activation."""
        if activation == "relu":
            return np.maximum(0.0, z)
        elif activation == "tanh":
            return np.tanh(z)
        elif activation == "leaky_relu":
            return np.where(z > 0, z, 0.01 * z)
        elif activation == "sigmoid":
            return self._sigmoid(z)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def _apply_activation_derivative(self, z: np.ndarray, activation: str) -> np.ndarray:
        """Calculate derivative of activation function given pre-activation z."""
        if activation == "relu":
            return np.where(z > 0, 1.0, 0.0)
        elif activation == "tanh":
            tanh_val = np.tanh(z)
            return 1.0 - tanh_val ** 2
        elif activation == "leaky_relu":
            return np.where(z > 0, 1.0, 0.01)
        elif activation == "sigmoid":
            sig = self._sigmoid(z)
            return sig * (1.0 - sig)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    # ---------------- Forward & Backward Propagation ---------------- #

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass through all layers.
        Returns the output probabilities and a cache of intermediate tensors.
        """
        cache: Dict[str, np.ndarray] = {"A0": X}
        A = X

        for l in range(1, self.num_layers):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            Z = np.dot(A, W) + b
            A = self._apply_activation(Z, self.hidden_activation)

            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        # Output layer with Sigmoid activation
        WL = self.params[f"W{self.num_layers}"]
        bL = self.params[f"b{self.num_layers}"]
        ZL = np.dot(A, WL) + bL
        AL = self._sigmoid(ZL)

        cache[f"Z{self.num_layers}"] = ZL
        cache[f"A{self.num_layers}"] = AL

        return AL, cache

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute Binary Cross-Entropy loss with optional L2 regularization.
        """
        m = y_true.shape[0]
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

        bce_loss = -np.mean(
            y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
        )

        # L2 Regularization term: (lambda / (2 * m)) * sum(||W||^2)
        l2_penalty = 0.0
        if self.l2_reg > 0:
            for l in range(1, self.num_layers + 1):
                l2_penalty += np.sum(np.square(self.params[f"W{l}"]))
            l2_penalty *= (self.l2_reg / (2.0 * m))

        return float(bce_loss + l2_penalty)

    def _backward(
        self, y_true: np.ndarray, cache: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradients of loss with respect to all parameters via backpropagation.
        """
        grads: Dict[str, np.ndarray] = {}
        m = y_true.shape[0]
        L = self.num_layers

        # Gradient at output layer (BCE loss + Sigmoid activation derivative simplifies cleanly):
        # dZ[L] = A[L] - Y
        AL = cache[f"A{L}"]
        dZL = AL - y_true

        grads[f"dW{L}"] = (np.dot(cache[f"A{L-1}"].T, dZL) / m) + (self.l2_reg / m) * self.params[f"W{L}"]
        grads[f"db{L}"] = np.sum(dZL, axis=0, keepdims=True) / m

        dZ_next = dZL

        # Propagate backward through hidden layers
        for l in range(L - 1, 0, -1):
            W_next = self.params[f"W{l+1}"]
            dA = np.dot(dZ_next, W_next.T)
            Z = cache[f"Z{l}"]
            dZ = dA * self._apply_activation_derivative(Z, self.hidden_activation)

            A_prev = cache[f"A{l-1}"]
            grads[f"dW{l}"] = (np.dot(A_prev.T, dZ) / m) + (self.l2_reg / m) * self.params[f"W{l}"]
            grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True) / m

            dZ_next = dZ

        return grads

    # ---------------- Parameter Updates (Optimizers) ---------------- #

    def _update_parameters(
        self,
        grads: Dict[str, np.ndarray],
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ) -> None:
        """
        Update model parameters using SGD, Momentum, or Adam.
        """
        self.t += 1

        for l in range(1, self.num_layers + 1):
            for p in ["W", "b"]:
                key = f"{p}{l}"
                grad_key = f"d{p}{l}"
                grad = grads[grad_key]

                if self.optimizer == "sgd":
                    self.params[key] -= self.learning_rate * grad

                elif self.optimizer == "momentum":
                    # Velocity update: v = beta * v + (1 - beta) * grad
                    self.v[grad_key] = beta1 * self.v[grad_key] + (1 - beta1) * grad
                    self.params[key] -= self.learning_rate * self.v[grad_key]

                elif self.optimizer == "adam":
                    # 1st moment vector (mean of gradients)
                    self.v[grad_key] = beta1 * self.v[grad_key] + (1.0 - beta1) * grad
                    # 2nd moment vector (uncentered variance)
                    self.s[grad_key] = beta2 * self.s[grad_key] + (1.0 - beta2) * (grad ** 2)

                    # Bias-corrected estimates
                    v_corrected = self.v[grad_key] / (1.0 - (beta1 ** self.t))
                    s_corrected = self.s[grad_key] / (1.0 - (beta2 ** self.t))

                    # Update step
                    step = self.learning_rate * v_corrected / (np.sqrt(s_corrected) + epsilon)
                    self.params[key] -= step

    # ---------------- Training & Inference ---------------- #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 500,
        batch_size: int = 32,
        verbose: bool = True
    ) -> "NeuralNetworkFromScratch":
        """
        Train the neural network using mini-batch gradient descent.
        """
        # Ensure y is 2D column vector (m, 1)
        y_train = y.reshape(-1, 1)
        y_val_col = y_val.reshape(-1, 1) if y_val is not None else None

        n_samples = X.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle training data at each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_train[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                _, cache = self._forward(X_batch)

                # Backward pass
                grads = self._backward(y_batch, cache)

                # Optimizer step
                self._update_parameters(grads)

            # Evaluate training metrics
            train_preds, _ = self._forward(X)
            train_loss = self.compute_loss(train_preds, y_train)
            train_acc = np.mean((train_preds >= 0.5) == y_train)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validation metrics
            if X_val is not None and y_val_col is not None:
                val_preds, _ = self._forward(X_val)
                val_loss = self.compute_loss(val_preds, y_val_col)
                val_acc = np.mean((val_preds >= 0.5) == y_val_col)

                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

            if verbose and (epoch % 50 == 0 or epoch == 1 or epoch == epochs):
                msg = f"Epoch {epoch:4d}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%"
                if X_val is not None:
                    msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
                print(msg)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class 1 probability."""
        probs, _ = self._forward(X)
        return probs.ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels (0 or 1)."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
