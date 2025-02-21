import numpy as np


def mse(y: np.ndarray, a: np.ndarray, n: int) -> float:
    return (1 / (2 * n)) * np.sum((y - a) ** 2)


def d_mse(y: np.ndarray, a: np.ndarray) -> np.ndarray:
    return -(y - a) / len(y)


def d2_mse(y: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.ones_like(a) / len(y)


def sigmoid(a: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-a))


def d_sigmoid(a: np.ndarray) -> np.ndarray:
    s = sigmoid(a)
    return s * (1 - s)


def d2_sigmoid(a: np.ndarray) -> np.ndarray:
    s = sigmoid(a)
    return s * (1 - s) * (1 - 2 * s)


def newton_hessian_backwards_pass(
    a_prev: np.ndarray,
    a: np.ndarray,
    w: np.ndarray,
    dC_da_next: np.ndarray,
    d2C_da2_next: np.ndarray,
    micro: float = 1e-6,
):
    """
    Backwards pass using Newton's method using a special Hessian matrix
    that's diagonal and positive-definite.

    Uses MSE for loss and Sigmoid for activation.

    # Parameters:
    - a_prev: previous layer activations
    - a: current layer activations
    - w: weights
    - dC_da_next: gradient of cost wrt activations of next layer
    - d2C_da2_next: second derivative of cost wrt activations of next layer

    # Returns:
    - Dw: weight updates
    """
    # Compute gradients wrt current activations (δ_i)
    dC_da = d_sigmoid(a) * np.sum(w * dC_da_next, axis=1)

    # Calculate the second derivative of cost wrt activations
    # term 1: f′(a_i)² · ∑_k (w²_{ki} · ∂²C/∂a²_k)
    # term 2: f"(a_i)² · ∑_k (w_{ki} · ∂C/∂a_k)
    # ∂²C/∂a²_i: term 1 - term 2
    d2C_da2_t1 = d_sigmoid(a) ** 2 * np.sum(w**2 * d2C_da2_next, axis=1)
    d2C_da2_t2 = d2_sigmoid(a) * np.sum(w * -dC_da_next, axis=1)
    d2C_da2 = d2C_da2_t1 - d2C_da2_t2

    # Calculate the derivative of cost wrt weights
    # ∂C/∂w_{ij} = δ_i ⋅ a_j
    dC_dw = dC_da[:, np.newaxis] * a_prev[np.newaxis, :]

    # Calculate the second derivative of cost wrt weights
    # ∂²C/∂w²_i: ∂²C/∂a²_i * f(a_j)²
    # In the paper, they use f(a_j)², but it most likely refers to pre-activation
    # values. So f(a_j)² is more likely to be σ(z_j)² and a_prev is already σ(z_j)²
    d2C_dw2 = d2C_da2[:, np.newaxis] * (a_prev**2)[np.newaxis, :]

    # Calculate the weight updates Δw_{ij}
    Dw = -dC_dw / (np.abs(d2C_dw2) + micro)

    return Dw
