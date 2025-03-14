import torch
import torch.nn as nn

from transformer import Encoder, Decoder


class ViT(nn.Module):
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
        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            enc_voc_size=enc_voc_size,
            drop_prob=drop_prob,
            num_layers=num_layers,
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
            device=device,
        )

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
    ) -> torch.Tensor:
        src = self.flatten(src)
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def flatten(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        n = h * w / (self.p**2)
        return x.reshape((b, n, self.p**2 * c))  # (batch x n x p²c)

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
