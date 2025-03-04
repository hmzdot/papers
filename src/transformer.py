import torch
import torch.nn as nn
from torch import Tensor
from relu import relu


def softmax(z: Tensor, dim=None) -> Tensor:
    z_exp = torch.exp(z)
    return z_exp / z_exp.sum(dim=dim, keepdim=True)


class LayerNorm:
    gamma: Tensor
    beta: Tensor
    eps: float

    def __init__(self, shape: tuple[int, ...], eps=1e-5) -> None:
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        # Parameters
        - x: (batch_size, seq_length, d_model) shaped Tensor
        """
        mean = x.mean(axis=-1, keepdim=True)
        std = x.std(axis=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / (std + self.eps)
        return normalized * self.gamma + self.beta


class SingleHeadAttention:
    W_Q: Tensor  # (d_model x d_k)
    W_K: Tensor  # (d_model x d_k)
    W_V: Tensor  # (d_model x d_v)
    W_O: Tensor  # (d_v x d_model)
    attention: "AttentionBlock"

    def __init__(self, d_model: int, d_k: int, d_v: int) -> None:
        sqrt_d_model = torch.sqrt(torch.tensor(d_model))
        sqrt_d_v = torch.sqrt(torch.tensor(d_v))
        self.W_Q = nn.Parameter(torch.randn(d_model, d_k)) / sqrt_d_model
        self.W_K = nn.Parameter(torch.randn(d_model, d_k)) / sqrt_d_model
        self.W_V = nn.Parameter(torch.randn(d_model, d_v)) / sqrt_d_model
        self.W_O = nn.Parameter(torch.randn(d_v, d_model)) / sqrt_d_v

        self.attention = AttentionBlock()

    def forward(self, x: Tensor) -> Tensor:
        Q = x @ self.W_Q  # Shape: (batch_size, seq_len, d_k)
        K = x @ self.W_K  # Shape: (batch_size, seq_len, d_k)
        V = x @ self.W_V  # Shape: (batch_size, seq_len, d_v)

        return self.attention(Q, K, V) @ self.W_O


class AttentionBlock:
    def __call__(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        """
        Performs attention function

        Attention(Q,K,V) = softmax(Q K^T / √d_k)V
        """
        d_k = torch.tensor(K.size(-1))  # Feature dimension of K
        return softmax((Q @ K.transpose(-1, -2)) / torch.sqrt(d_k), dim=-1) @ V


class FFN:
    W1: Tensor  # (d_model x d_ff)
    W2: Tensor  # (d_ff x d_model)
    b1: Tensor  # (d_ff x 1)
    b2: Tensor  # (d_model x 1)

    def __init__(self, d_model: int, d_ff: int) -> None:
        self.W1 = nn.Parameter(torch.randn((d_model, d_ff)))
        self.W2 = nn.Parameter(torch.randn((d_ff, d_model)))
        self.b1 = nn.Parameter(torch.zeros((d_ff,)))
        self.b2 = nn.Parameter(torch.zeros((d_model,)))

    def forward(self, x: Tensor):
        """
        # Parameters:
        - x: (batch_size, seq_length, d_model) shaped Tensor
        """
        return relu(x @ self.W1 + self.b1) @ self.W2 + self.b2
