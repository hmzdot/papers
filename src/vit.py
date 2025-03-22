import torch
import torch.nn as nn

from transformer import Encoder, Decoder, TokenEmbedding


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "Image not divisible by patch"

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size**2 * in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        # Parameters:
        - x: [batch, channels, height, width]
        # Returns:
        - output: [batch, num_patches, embed_dim]
        """
        b = x.size(0)
        p = self.patch_size

        x = x.unfold(2, p, p).unfold(3, p, p)  # [b, c, h//p, w//p, p, p]
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(
            b, self.num_patches, -1
        )  # [b, num_patches, p²c]
        x = self.projection(x)  # [b, num_patches, embed_dim]

        return x


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
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=d_model,
        )
        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            enc_voc_size=enc_voc_size,
            drop_prob=drop_prob,
            num_layers=num_layers,
            device=device,
            embedding=self.patch_embedding,
        )

        self.token_embedding = TokenEmbedding(
            vocab_size=dec_voc_size,
            d_model=d_model,
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
            embedding=self.token_embedding,
        )

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
    ) -> torch.Tensor:
        """
        # Parameters:
        - src: [batch_size, channels, height, width]
        - trg: [batch_size, trg_len]
        # Returns:
        - output: [batch_size, trg_len, trg_vocab_size]
        """
        src_mask = self.make_src_mask(src)  # [b, 1, 1, n]
        trg_mask = self.make_trg_mask(trg)  # [b, 1, trg_len, trg_len]
        enc_src = self.encoder(src, src_mask)  # [b, n, d_model]
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output  # [batch_size, trg_len, trg_vocab_size]

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
