import torch
from torch import Tensor, nn

from type.gpt_configuration import GPTConfiguration


class FeedForward(nn.Module):
    """Feed-forward layer."""

    def __init__(self, cfg: GPTConfiguration):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


if __name__ == "__main__":
    cfg = {"emb_dim": 16}

    feed_forward_layer = FeedForward(cfg)
    x = torch.randn(2, 16)  # Batch of 2 samples, each of dimension 16
    output = feed_forward_layer(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output:", output)
