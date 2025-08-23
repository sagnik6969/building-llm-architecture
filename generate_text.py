import tiktoken
import torch

from gpt import GPT
from llm_pretraining_loop import generate_and_print_sample

model = GPT(
    {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-key-value bias
    }
)

model.load_state_dict(
    torch.load("checkpoint.pt", map_location=torch.device("cpu"))["model"]
)

tokenizer = tiktoken.get_encoding("gpt2")

generate_and_print_sample(model, tokenizer, "cpu", "Can")
