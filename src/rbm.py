import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """σ(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-x))


class RBM:
    """Restriced Boltzman Machine"""

    num_visible: int
    """Number of visible layers. Represents input features"""

    num_hidden: int
    """Number of hidden layers. Captures latent features (?)"""
    a: np.ndarray

    """Visible layer biases"""

    b: np.ndarray
    """Hidden layer biases"""

    W: np.ndarray
    """Weights"""

    batch_size: int

    def __init__(self, num_visible: int, num_hidden: int, batch_size=32):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.batch_size = batch_size

        self.W = np.random.normal(0, 0.01, (num_visible, num_hidden))

        self.a = np.zeros((num_visible, 1))
        self.b = np.zeros((num_hidden, 1))

    def energy(self, v: np.ndarray, h: np.ndarray) -> float:
        """
        # Parameters
        - v: Visible layer units (shaped n x 1)
        - h: Hidden layer units (shaped h x 1)

        Energy function is defined as:
        E(v,h) = -∑ᵢ(aᵢvᵢ) - ∑ⱼ(bⱼhⱼ) - ∑ᵢⱼ(vᵢWᵢⱼhⱼ)
        """
        return -(
            np.dot(v, self.a) + np.dot(h, self.b) + np.dot(np.dot(v.T, self.W), h)
        ).item()

    def sample_hidden(self, v: np.ndarray) -> np.ndarray:
        """
        Given visible units, sample hidden units using Bernoulli
        Probability of visible, given hidden is defined as:

        P(h_j = 1 | v) = σ(b_j + ∑_i(v_i W_ij))

        where σ is the sigmoid function
        """
        prob_h_given_v = sigmoid(self.b + np.dot(self.W, v))
        return np.random.binomial(1, prob_h_given_v).astype(int)

    def sample_visible(self, h: np.ndarray) -> np.ndarray:
        """
        Given hidden units, sample visible units using Bernoulli
        Probability of hidden, given visible is defined as:

        P(v_i = 1 | h) = σ(a_i + ∑_j h_jW_ij)

        where σ is the sigmoid function
        """
        prob_v_given_h = sigmoid(self.a + np.dot(self.W, h))
        return np.random.binomial(1, prob_v_given_h).astype(int)

    def gibbs_sampling(self, v: np.ndarray, k=1) -> np.ndarray:
        """
        Performs k-step Gibbs sampling

        # Parameters
        - v: Visible units

        Gibbs sampling is a Markov Chain Monte Carlo (MCMC) method used to sample
        from a joint probability distribution when direct sampling is infeasible
        """

        for _ in range(k):
            h = self.sample_hidden(v)
            v = self.sample_visible(h)
        return v

    def train(self, data: np.ndarray, epochs=10, lr=0.1) -> float:
        """
        Contrastive Divergence (CD-1) training

        # Parameters
        - data: Input, shaped (num_samples, num_visible)
        - epochs: Number of epoches
        - lr: Learning rate
        """

        num_batches = data.shape[0] // self.batch_size
        batches = data[: num_batches * self.batch_size].reshape(
            num_batches, self.batch_size, self.num_visible
        )
        for _ in range(epochs):
            for i in range(batches.shape[0]):
                # Positive phase
                v = batches[i, ...].T  # (n x batch_size)
                h = self.sample_hidden(v)  # (h x batch_size)
                vhT = v.dot(h.T)  # (n x h)

                # Negative phase
                vp = self.gibbs_sampling(v)  # (n x batch_size)
                hp = self.sample_hidden(vp)  # (h x batch_size)
                vphpT = vp.dot(hp.T)  # (n x h)

                self.W += lr * (vhT - vphpT) / self.batch_size
                self.a += lr * np.sum(v - vp, axis=1, keepdims=True)
                self.b += lr * np.sum(h - hp, axis=1, keepdims=True)
