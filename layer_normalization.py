from torch import nn,Tensor
import torch

class LayerNorm(nn.Module):
    """Layer normalization layer."""
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x:Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
if __name__ == "__main__":
    # Set PyTorch to not use scientific notation
    torch.set_printoptions(sci_mode=False)
    layer_norm = LayerNorm(emb_dim=512)
    x = torch.randn(32, 512)  # Batch of 32 samples, each of dimension 512
    output:Tensor = layer_norm(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output variance:", output.var(dim=-1))
    print("Output mean:", output.mean(dim=-1))
    # print(output)