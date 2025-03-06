import math
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from relu import relu


def softmax(z: Tensor, dim=None) -> Tensor:
    z_max = z.max(dim=dim, keepdim=True).values
    z_stable = z - z_max
    z_exp = torch.exp(z_stable)
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
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        xavier_uniform(self.W_Q.weight, fan_in=d_model, fan_out=d_model)
        xavier_uniform(self.W_K.weight, fan_in=d_model, fan_out=d_model)
        xavier_uniform(self.W_V.weight, fan_in=d_model, fan_out=d_model)
        xavier_uniform(self.W_O.weight, fan_in=d_model, fan_out=d_model)

        self.attention = AttentionBlock()

    def forward(
        self,
        Q: Tensor,
        K: Union[Tensor, None] = None,
        V: Union[Tensor, None] = None,
        mask=None,
    ) -> Tensor:
        if K is None:
            K = Q
        if V is None:
            V = K

        batch_size, _, _ = Q.shape
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output = self.attention(Q, K, V, mask)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        return self.W_O(attn_output)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.W1 = nn.Parameter(torch.empty((d_model, d_ff)))
        self.W2 = nn.Parameter(torch.empty((d_ff, d_model)))
        self.b1 = nn.Parameter(torch.zeros((d_ff,)))
        self.b2 = nn.Parameter(torch.zeros((d_model,)))

        xavier_uniform(self.W1, fan_in=d_model, fan_out=d_ff)
        xavier_uniform(self.W2, fan_in=d_ff, fan_out=d_model)

    def forward(self, x: Tensor) -> Tensor:
        return relu(x @ self.W1 + self.b1) @ self.W2 + self.b2


class PositionalEncoding(nn.Module):
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

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
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
    def __init__(self, n: int, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(n)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, num_heads)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_model, d_ff)
        self.norm1 = LayerNorm((d_model,))
        self.norm2 = LayerNorm((d_model,))
        self.norm3 = LayerNorm((d_model,))

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        # Create dynamic mask based on sequence length
        _, seq_length, _ = x.shape
        mask = torch.triu(
            torch.ones(seq_length, seq_length),
            diagonal=1,
            device=x.device,
        ).bool()

        masked_attn_out = self.masked_attn(Q=x, K=x, V=x, mask=mask)
        x = self.norm1(x + masked_attn_out)

        enc_dec_attn_out = self.attn(Q=x, K=memory, V=memory)
        x = self.norm2(x + enc_dec_attn_out)

        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


class Decoder(nn.Module):
    def __init__(self, n: int, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff) for _ in range(n)]
        )

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_length: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        # Encoder
        src_emb = self.embedding(src)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb)

        # Decoder
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.decoder(tgt_emb, memory)

        return self.fc_out(output)
