import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function"""
    return np.maximum(0, x)


def d_relu(x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU activation function
    d/dx ReLU(x) = 0 if x <= 0, 1 if x > 0
    """
    return np.where(x > 0, 1, 0)


def nrelu(x: np.ndarray, sigma=0.1) -> np.ndarray:
    """
    Noisy ReLU activation function. Sometimes used in RBMs to introduce more
    randomness into the model.

    NReLU(x) = max(0, x + ε)

    where ε ~ N(0, σ^2)
    """
    noise = np.random.normal(0, sigma, size=x.shape)
    return np.maximum(0, x + noise)


def d_nrelu(x: np.ndarray, noise: float) -> np.ndarray:
    """
    Derivative of Noisy ReLU activation function
    d/dx NReLU(x) = 0 if x + ε <= 0, 1 if x + ε > 0
    """
    return np.where(x + noise > 0, 1, 0)
