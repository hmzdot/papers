import numpy as np


def nll(y: np.ndarray, x: np.ndarray) -> float:
    """
    Calculate the Negative Log Likelihood (NLL) loss.

    # Parameters
    - y: np.ndarray: Correct label as a one-hot encoded vector or a scalar
    - x: np.ndarray: Model's output as a probability distribution vector

    # Returns
    - float: The negative log likelihood

    Given a probabilistic model that outputs a likelihood P(y|x;θ) for a
    given data point (x,y), the log likelihood is defined as:

    log(P(y|x;θ))

    > Note: P(y|x;θ) is the probability of the correct label y given the input x
    and the model parameters θ. Semicolon is used to denote that θ is not
    a random variable.

    Since we want to maximize the probability of the correct label, the objective
    is to maximize the log-likelihood. But optimization frameworks typically
    minimize losses, so we take the negative of the log-likelihood:

    NLL = -log(P(y|x;θ))

    This is the negative log likelihood loss.
    """
    if y.ndim > 0:
        class_index = np.argmax(y)
    else:
        class_index = int(y)

    # Clip the probability to avoid log(0) which is undefined
    epsilon = 1e-10
    probability = np.clip(x[class_index], epsilon, 1.0)

    return -np.log(probability)
