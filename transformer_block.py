import torch
from torch import nn, Tensor
from type.gpt_configuration import GPTConfiguration
from layer_normalization import LayerNorm
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfiguration):
        super().__init__()
        self.layer_norm1 = LayerNorm(cfg["emb_dim"])
        self.multi_head_attention = MultiHeadAttention(
            cfg["emb_dim"],
            cfg["emb_dim"],
            cfg["context_length"],
            cfg["drop_rate"],
            cfg["n_heads"],
        )
        self.dropout1 = nn.Dropout(cfg["drop_rate"])
        self.layer_norm2 = LayerNorm(cfg["emb_dim"])
        self.feed_forward = FeedForward(cfg)
        self.dropout2 = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: Tensor) -> Tensor:
        layer_norm_1_out = self.layer_norm1(x)
        multi_head_attention_out = self.multi_head_attention(layer_norm_1_out)
        drop_out_output = self.dropout1(multi_head_attention_out)
        shortcut_1_output = x + drop_out_output
        layer_norm_2_out = self.layer_norm2(shortcut_1_output)
        feed_forward_out = self.feed_forward(layer_norm_2_out)
        drop_out_output_2 = self.dropout2(feed_forward_out)
        shortcut_2_output = shortcut_1_output + drop_out_output_2

        return shortcut_2_output
    

if __name__ == "__main__":
    cfg = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
   }

    model = TransformerBlock(cfg)
    input_tensor = torch.randn(1, 1024, 768)  # (batch_size, context_length, emb_dim)
    output: Tensor = model(input_tensor)
    print(output.shape)
