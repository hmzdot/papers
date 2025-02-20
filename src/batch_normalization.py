import numpy as np


def batchnorm_forward(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Forward pass of BatchNorm"""

    """
    BatchNorm function is defined as:

    $$ y = \\gamma \\cdot \\hat{x}^k + \\beta $$

    where:
    - $\\gamma$ and $\\beta$ are trainable parameters that keeps representation
        power of the network after normalization
    - $\\hat{x}$ is the normalized input and defined as:
    
    $$ \\hat{x} = \\frac{ x - E[x] }{ \\sqrt{ \\sigma^2(x) } } $$

    where:
    - E[x] is expected value of x, ie. mean of x
    - \\sqrt{ \\sigma^2(x) } is standard deviation of x
    
    """

    mean = np.mean(x, axis=0, keepdim=True)
    var = np.var(x, axis=0, keepdim=True)
    x_hat = (x - mean) * np.sqrt(var + epsilon)
    return gamma * x_hat + beta


def batchnorm_backward(
    dl_dy: np.ndarray,
    x: np.ndarray,
    gamma: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Backward pass of BatchNorm"""
    m, _ = x.shape  # Get batch size

    mean = np.mean(x, axis=0, keepdim=True)
    var = np.var(x, axis=0, keepdim=True)
    std_inv = 1.0 / np.sqrt(var + epsilon)
    x_hat = (x - mean) / std_inv

    dl_dgamma = np.sum(dl_dy * x_hat, axis=0)
    dl_dbeta = np.sum(dl_dy, axis=0)
    dl_dxhat = dl_dy * gamma
    dl_dvar = np.sum(dl_dxhat * (x - mean) * -0.5 * std_inv**3, axis=0)
    dl_dmean = (
        np.sum(dl_dxhat * -std_inv, axis=0)
        + dl_dvar * np.sum(-2 * (x - mean), axis=0) / m
    )
    dl_dx = dl_dxhat * std_inv + dl_dvar * 2 * (x - mean) / m + dl_dmean / m

    return dl_dx, dl_dgamma, dl_dbeta
