"""Shared single-pass prediction for DNS TRM."""
import torch
from single_state import SingleStateModel


@torch.no_grad()
def predict_tokens(model, embedder, tokens):
    if tokens.ndim not in (2, 3):
        raise ValueError("Expected token IDs [B,T] or continuous features [B,T,D]")
    batch, length = tokens.shape[:2]
    if isinstance(model, SingleStateModel):
        logits, _ = model(embedder(tokens))
        return logits.argmax(dim=-1)
    y = torch.zeros(batch, length, model.d, device=tokens.device)
    z = torch.zeros_like(y)
    x = embedder(tokens)
    _, logits, _ = model.deep_recursion(x, y, z)
    return logits.argmax(dim=-1)
