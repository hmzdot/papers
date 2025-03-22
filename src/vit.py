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
        num_classes=0,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "Image not divisible by patch"

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size**2 * in_channels, embed_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + (1 if num_classes else 0), embed_dim)
        )
        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, embed_dim)) if num_classes > 0 else None
        )

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

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        x += self.pos_embedding
        return x


class ViT(nn.Module):
    def __init__(
        self,
        src_pad_idx: int = 0,
        trg_pad_idx: int = 0,
        trg_sos_idx: int = 1,
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
        num_classes: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.num_classes = num_classes

        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=d_model,
            num_classes=num_classes,
        )
        self.encoder = Encoder(
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            num_heads=num_heads,
            num_layers=num_layers,
            drop_prob=drop_prob,
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

        if self.num_classes > 0:
            self.cls_head = nn.Linear(d_model, num_classes)
        else:
            self.cls_head = None

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

        if self.num_classes > 0:
            cls_embedding = enc_src[:, 0, :]  # Embeddings of [CLS] token
            cls_logits = self.cls_head(cls_embedding)
        else:
            cls_logits = None

        return output, cls_logits  # [batch_size, trg_len, trg_vocab_size]

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        # Parameters:
        - src: [batch_size, channels, height, width]

        # Returns:
        - Mask [batch_size, 1, 1, num_patches + 1], 1 to attend to, 0 to ignore
        """
        batch_size, _, _, _ = src.shape
        return torch.ones(
            batch_size,
            1,
            1,
            self.patch_embedding.num_patches + (1 if self.num_classes else 0),
            device=self.device,
        )

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


def main():
    # Initialize the model
    model = ViT(
        img_size=224,
        patch_size=16,
        in_channels=3,
        d_model=512,
        num_heads=8,
        max_len=1024,
        ffn_hidden=2048,
        num_layers=6,
        drop_prob=0.1,
        num_classes=2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Create a dummy image tensor
    image = torch.randn(1, 3, 224, 224)

    # Create a dummy target tensor (e.g., a sequence of token indices)
    target = torch.randint(0, 5000, (1, 10))  # Assuming dec_voc_size=5000

    # Forward pass
    output, cls_head = model(image, target)

    # Output shape should be [batch_size, trg_len, dec_voc_size]
    print(output.shape)  # Should be torch.Size([1, 10, 5000])
    print(cls_head)


if __name__ == "__main__":
    main()
