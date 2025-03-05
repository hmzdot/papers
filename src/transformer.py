import math
import torch
import torch.nn as nn
from torch import Tensor
from relu import relu


def softmax(z: Tensor, dim=None) -> Tensor:
    z_exp = torch.exp(z)
    return z_exp / z_exp.sum(dim=dim, keepdim=True)


def xavier_uniform(x: Tensor, fan_in: int, fan_out: int):
    """
    Performs uniform Xavier initialization

    # Parameters
    - x: Input tensor
    - fan_in: Input parameter count (n_in)
    - fan_out: Output parameter count (n_out)

    Uniform Xavier initialization is defined as:

    W_0 = U( -√(6 / (n_in + n_out)), √(6 / (n_in + n_out)) )

    where U is the uniform function
    """
    a = math.sqrt(6 / (fan_in + fan_out))
    with torch.no_grad():
        return x.uniform_(-a, a)


class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Create a binary mask using Bernoulli distribution
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p))
            return x * mask / (1 - self.p)
        return x


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
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout = Dropout(p=dropout_rate)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask=None) -> Tensor:
        """
        Performs attention function

        Attention(Q,K,V) = softmax(Q K^T / √d_k)V
        """
        d_k = K.size(-1)  # Feature dimension of K
        attn_scores = (Q @ K.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -1e9)
        attn_weights = softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mask=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.mask = mask

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        xavier_uniform(self.W_Q.weight)
        xavier_uniform(self.W_K.weight)
        xavier_uniform(self.W_V.weight)
        xavier_uniform(self.W_O.weight)

        self.attention = AttentionBlock()

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        Q = self.W_Q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        attn_output = self.attention(Q, K, V, self.mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(attn_output)


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
        self.attn = MultiHeadAttention(d_model, d_k, d_v)
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


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        mask = torch.triu(torch.ones(d_k, d_k), diagonal=1)
        self.masked_attn = MultiHeadAttention(d_model, d_k, d_v, mask=mask)
        self.attn = MultiHeadAttention(d_model, d_k, d_v)
        self.ffn = FFN(d_model, d_ff)
        self.norm1 = LayerNorm((d_model,))
        self.norm2 = LayerNorm((d_model,))

    def forward(self, x: Tensor) -> Tensor:
        masked_attn_out = self.masked_attn(x)
        x = self.norm1(x + masked_attn_out)

        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class Decoder(nn.Module):
    def __init__(self, n: int, d_model: int, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_k, d_v, d_ff) for _ in range(n)]
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
