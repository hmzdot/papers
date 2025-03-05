import math
import torch
import torch.nn as nn
from torch import Tensor
from relu import relu


def softmax(z: Tensor, dim=None) -> Tensor:
    z_exp = torch.exp(z)
    return z_exp / z_exp.sum(dim=dim, keepdim=True)


class LayerNorm(nn.Module):
    def __init__(self, shape: tuple[int, ...], eps=1e-5) -> None:
        super().__init__()
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


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        """
        Performs attention function

        Attention(Q,K,V) = softmax(Q K^T / √d_k)V
        """
        d_k = torch.tensor(K.size(-1))  # Feature dimension of K
        return softmax((Q @ K.transpose(-1, -2)) / torch.sqrt(d_k), dim=-1) @ V


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int) -> None:
        super().__init__()
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


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
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


class PositionalEncoding(nn.Module):
    """Dimensionality of the model"""

    def __init__(self, d_model: int, max_seq_length=5000):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_length, d_model)
        pos = torch.arange(0, max_seq_length).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000)) / d_model
        )

        # PE(pos,2i) = sin(pos/10_000^(2i/d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)

        # PE(pos,2i+1) = cos(pos/10_000^(2i/d_model))
        pe[:, 1::2] = torch.cos(pos * div_term)

        # Register as buffer (not a trainable parameter)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.attn = SingleHeadAttention(d_model, d_k, d_v)
        self.ffn = FFN(d_model, d_ff)
        self.norm1 = LayerNorm((d_model,))
        self.norm2 = LayerNorm((d_model,))

    def forward(self, x: Tensor) -> Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class Encoder(nn.Module):
    def __init__(self, n: int, d_model: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_k, d_v, d_ff) for _ in range(n)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        d_k: int,
        d_v: int,
        d_ff: int,
        max_seq_length: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.encoder = Encoder(num_layers, d_model, d_k, d_v, d_ff)
