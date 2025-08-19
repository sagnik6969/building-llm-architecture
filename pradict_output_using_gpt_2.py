import tiktoken
import torch
from gpt import GPT
import logging



def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probabilities, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

if __name__ == "__main__":
    mps_device = torch.device("mps")
    tokenizer = tiktoken.get_encoding("gpt2")
    logging.info("Tokenizer loaded")

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
    model.to(mps_device)  # Move model to MPS device
    logging.info("Model loaded")
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    logging.info("Encoded input text")
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #A
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval() #A
    out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=cfg["context_length"]
    )
    logging.info("Generated output")
    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)