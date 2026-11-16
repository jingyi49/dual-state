"""Single-state controls for the four-layer-per-state WavLM TRM baseline."""
import torch
from torch import nn
from trm import TinyTransformer


class SingleStateModel(nn.Module):
    """One residual state update; shallow=4 layers, deep=8 untied layers.

    Keep the baseline's output/confidence heads and residual normalization.
    No y/z concatenations or second state are used.
    """
    def __init__(self, d_model, vocab_size, *, n_layers, n_heads=8, dropout=0.1):
        super().__init__()
        self.d = d_model
        self.input_projection = nn.Linear(d_model, d_model)
        self.net = TinyTransformer(d_model, n_heads, n_layers, dropout)
        self.ln = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
        self.q_pool = nn.AdaptiveAvgPool1d(1)
        self.q_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                   nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.input_projection(x)
        h = self.ln(self.net(h) + h)
        logits = self.output_head(h)
        q = torch.sigmoid(self.q_mlp(self.q_pool(h.transpose(1, 2)).squeeze(-1)).squeeze(-1))
        return logits, q
