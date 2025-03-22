import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None) -> torch.Tensor:
        # [batch_size, head, length, d_tensor]
        d_k = K.size(-1)
        score = (Q @ K.transpose(-1, -2)) / math.sqrt(d_k)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        score = self.softmax(score)
        return score @ V, score


class MSA(nn.Module):
    """Multi-head Self Attention"""

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        nn.init.xavier_uniform_(self.W_Q.weight, gain=1)
        nn.init.xavier_uniform_(self.W_K.weight, gain=1)
        nn.init.xavier_uniform_(self.W_V.weight, gain=1)
        nn.init.xavier_uniform_(self.W_O.weight, gain=1)

        self.attention = ScaledDotProductAttention()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask=None,
    ) -> torch.Tensor:
        """
        # Parameters
        - Q: (batch x seq_len x d_k)
        - K: (batch x seq_len x d_k)
        - V: (batch x seq_len x d_v)
        """
        # Reshape Q, K, V to (batch x num_heads x seq_len x d_k/d_v)
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)
        Q, K, V = self.split(Q), self.split(K), self.split(V)

        out, _ = self.attention(Q, K, V, mask)
        out = self.concat(out)
        out = self.W_O(out)

        return out

    def split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = tensor.size()
        d_tensor = d_model // self.num_heads
        return tensor.view(batch_size, seq_len, self.num_heads, d_tensor).transpose(
            1, 2
        )

    def concat(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        return tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device: torch.device):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.device = device

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        two_i = torch.arange(0, d_model, step=2, device=device)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (two_i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (two_i / d_model)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :]


class TokenEmbedding(nn.Embedding):
    def __init__(self, d_model: int, vocab_size: int, device: torch.device):
        super().__init__(vocab_size, d_model, device=device)


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        drop_prob: float,
        device: torch.device,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(d_model, vocab_size, device)
        self.positional_encoding = PositionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_emb = self.token_embedding(x)
        pos_emb = self.positional_encoding(x)
        return self.dropout(token_emb + pos_emb)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, drop_prob: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        drop_prob: float,
    ):
        super().__init__()
        self.attn = MSA(num_heads, d_model)
        self.norm1 = nn.LayerNorm((d_model,))
        self.dropout1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, drop_prob)
        self.norm2 = nn.LayerNorm((d_model,))
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(Q=x, K=x, V=x, mask=src_mask)
        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        ffn_out = self.dropout2(ffn_out)
        x = self.norm2(x + ffn_out)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        num_layers: int,
        drop_prob: float,
        embedding: nn.Module,
    ):
        super().__init__()
        self.embedding = embedding
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=ffn_hidden,
                    drop_prob=drop_prob,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, drop_prob: float):
        super().__init__()
        self.self_attn = MSA(num_heads, d_model)
        self.norm1 = nn.LayerNorm((d_model,))
        self.dropout1 = nn.Dropout(drop_prob)

        self.enc_attn = MSA(num_heads, d_model)
        self.norm2 = nn.LayerNorm((d_model,))
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, drop_prob)
        self.norm3 = nn.LayerNorm((d_model,))
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(
        self,
        dec_in: torch.Tensor,
        enc_out: torch.Tensor,
        trg_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = dec_in

        self_attn_out = self.self_attn(Q=x, K=x, V=x, mask=trg_mask)
        self_attn_out = self.dropout1(self_attn_out)
        x = self.norm1(self_attn_out + x)

        if enc_out is not None:
            enc_attn_out = self.enc_attn(Q=x, K=enc_out, V=enc_out, mask=src_mask)
            enc_attn_out = self.dropout2(enc_attn_out)
            x = self.norm2(enc_attn_out + x)

        ffn_out = self.ffn(x)
        ffn_out = self.dropout3(ffn_out)
        x = self.norm3(ffn_out + x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        dec_voc_size: int,
        max_len: int,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        num_layers: int,
        drop_prob: float,
        embedding: nn.Module,
        device: torch.device,
    ):
        super().__init__()
        self.embedding = embedding
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=ffn_hidden,
                    drop_prob=drop_prob,
                )
                for _ in range(num_layers)
            ]
        )
        self.inner = nn.Linear(d_model, dec_voc_size)

    def forward(
        self,
        trg: torch.Tensor,
        enc_out: torch.Tensor,
        trg_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        trg = self.embedding(trg)

        for layer in self.layers:
            trg = layer(trg, enc_out, trg_mask, src_mask)

        return self.inner(trg)


class Transformer(nn.Module):
    def __init__(
        self,
        src_pad_idx: int = 0,
        trg_pad_idx: int = 0,
        trg_sos_idx: int = 1,
        enc_voc_size: int = 5000,
        dec_voc_size: int = 5000,
        d_model: int = 512,
        num_heads: int = 8,
        max_len: int = 1024,
        ffn_hidden: int = 2048,
        num_layers: int = 6,
        drop_prob: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder_embedding = TransformerEmbedding(
            vocab_size=enc_voc_size,
            d_model=d_model,
            max_len=max_len,
            drop_prob=drop_prob,
            device=device,
        )

        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            enc_voc_size=enc_voc_size,
            drop_prob=drop_prob,
            num_layers=num_layers,
            embedding=self.encoder_embedding,
            device=device,
        )

        self.decoder_embedding = TransformerEmbedding(
            vocab_size=dec_voc_size,
            d_model=d_model,
            max_len=max_len,
            drop_prob=drop_prob,
            device=device,
        )
        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            dec_voc_size=dec_voc_size,
            drop_prob=drop_prob,
            num_layers=num_layers,
            embedding=self.decoder_embedding,
            device=device,
        )

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
    ) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        # Parameters:
        - src: [batch_size, src_len]

        # Returns:
        - Mask [batch_size, 1, 1, src_len], 1 to attend to, 0 to ignore
        """
        padding_mask = src != self.src_pad_idx  # [batch_size, src_len]
        return padding_mask.unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        """
        # Parameters:
        - trg: [batch_size, trg_len]

        # Returns:
        - Mask [batch_size, 1, trg_len, trg_len], 1 to attend to, 0 to ignore
        """
        _, trg_len = trg.shape

        padding_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        look_ahead_mask = torch.tril(
            torch.ones(trg_len, trg_len, device=self.device, dtype=torch.bool)
        )
        combined_mask = padding_mask & look_ahead_mask

        return combined_mask
