import numpy as np


def energy(
    v: np.ndarray,
    a: np.ndarray,
    h: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
) -> float:
    """
    # Parameters
    - v: Visible layer units (shaped n x 1)
    - a: Visible layer biases (shaped n x 1)
    - h: Hidden layer units (shaped h x 1)
    - b: Hidden layer biases (shaped h x 1)
    - W: Weights between visible and hidden units (shaped n x h)
    """
    return -(np.dot(v, a) + np.dot(h, b) + np.dot(np.dot(v.T, W), h)).item()
