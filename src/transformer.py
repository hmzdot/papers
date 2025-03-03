import torch
from torch import Tensor
from relu import relu


def softmax(z: torch.Tensor):
    z_exp = torch.exp(z)
    return z_exp / z_exp.sum()


class LayerNorm:
    gamma: Tensor
    beta: Tensor
    eps: float

    def __init__(self, shape: tuple[int, ...], eps=1e-5) -> None:
        self.gamma = torch.ones(shape)
        self.beta = torch.zeros(shape)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=1, keepdim=True)
        std = x.std(axis=1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        return normalized * self.gamma + self.beta


class FFN:
    W1: Tensor  # (d_model x d_ff)
    W2: Tensor  # (d_ff x d_model)
    b1: Tensor  # (d_ff x 1)
    b2: Tensor  # (d_model x 1)

    def __init__(self, d_model: int, d_ff: int) -> None:
        self.W1 = torch.randn((d_model, d_ff))
        self.W2 = torch.randn((d_ff, d_model))
        self.b1 = torch.zeros((d_ff,))
        self.b2 = torch.zeros((d_model,))

    def forward(self, x: Tensor):
        """
        # Parameters:
        - x: (batch_size, seq_length, d_model) shaped Tensor
        """
        return relu(x @ self.W1 + self.b1) @ self.W2 + self.b2
