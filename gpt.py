import torch
from torch import Tensor, nn

from layer_normalization import LayerNorm
from transformer_block import TransformerBlock
from type.gpt_configuration import GPTConfiguration


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfiguration):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout_embedding = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_normalization = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, context_length = x.shape
        token_embeddings = self.token_embedding(x)  # Shape: (batch_size, context_length
        positional_embeddings = self.position_embedding(
            torch.arange(context_length, device=x.device)
        )

        x = token_embeddings + positional_embeddings
        x = self.dropout_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_normalization(x)
        logits = self.out_head(x)

        return logits


if __name__ == "__main__":
    cfg = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
    }
    model = GPT(cfg)
    inputs: Tensor = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]))
    outputs: Tensor = model(inputs)
    print(outputs.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    total_size_bytes = total_params * 4  # A
    total_size_mb = total_size_bytes / (1024 * 1024)  # B
    print(f"Total size of the model: {total_size_mb:.2f} MB")
