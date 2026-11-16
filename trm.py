#!/usr/bin/env python3

# ==============================================================================
# Copyright 2025 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tiny Recursion Models."""

import torch
import torch.nn as nn


# ----------------------------
# Positional embeddings (learned)
# ----------------------------
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: [B,T,D]
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return x + self.pe(pos)


# ----------------------------
# Input embedder (tokens or features)
# ----------------------------
class InputEmbedder(nn.Module):
    def __init__(self, vocab_size=None, d_model=256, in_dim=None, max_len=4096):
        super().__init__()
        if vocab_size is not None:
            self.emb = nn.Embedding(vocab_size, d_model)
            self.proj = None
        else:
            assert in_dim is not None
            self.emb = None
            self.proj = nn.Linear(in_dim, d_model)
        self.pos = LearnedPositionalEncoding(max_len=max_len, d_model=d_model)

    def forward(self, x):
        if self.emb is not None:
            h = self.emb(x)  # [B,T,D]
        else:
            h = self.proj(x)  # [B,T,D]
        return self.pos(h)


# ----------------------------
# 4× Transformer "net"
# ----------------------------
class TinyTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=4, dropout=0.1, causal=False):
        super().__init__()
        self.causal = causal
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pre_ln = nn.LayerNorm(d_model)

    def _causal_mask(self, T, device):
        # Upper-triangular mask with -inf above diagonal
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, h):
        """
        h: [B,T,D]
        returns: [B,T,D]
        """
        T = h.size(1)
        mask = self._causal_mask(T, h.device) if self.causal else None
        h = self.blocks(self.pre_ln(h), mask=mask)
        return h


# ----------------------------
# Recursive model
# ----------------------------
class TinyRecursiveModel(nn.Module):
    """A single latent update followed by token and confidence heads."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        *,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        causal: bool = False,
        share_nets: bool = False,
    ):
        super().__init__()
        self.d = d_model
        self.vocab_size = vocab_size

        # Fuse projections (concat features -> d_model)
        self.fuse_latent = nn.Linear(d_model * 3, d_model)  # [x,y,z] -> h_z
        self.fuse_answer = nn.Linear(d_model * 2, d_model)  # [y,z]   -> h_y

        # 4× Transformer blocks (either shared or separate)
        self.net_z = TinyTransformer(d_model, n_heads, n_layers, dropout, causal)
        self.net_y = (
            self.net_z
            if share_nets
            else TinyTransformer(d_model, n_heads, n_layers, dropout, causal)
        )

        # Heads on y
        self.output_head = nn.Linear(d_model, vocab_size)  # per-timestep logits [B,T,V]
        self.q_pool = nn.AdaptiveAvgPool1d(1)  # pool over time (channels-first)
        self.q_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),  # sigmoid later
        )

        self.ln_y = nn.LayerNorm(d_model)
        self.ln_z = nn.LayerNorm(d_model)

    @staticmethod
    def _catf(*tensors):
        # concat on feature dim: all [B,T,D]
        return torch.cat(tensors, dim=-1)

    def latent_recursion(self, x, y, z):
        """Apply the fixed single latent/answer update."""
        hz = self.fuse_latent(self._catf(x, y, z))
        z = self.ln_z(self.net_z(hz) + hz)

        hy = self.fuse_answer(self._catf(y, z))
        y = self.ln_y(self.net_y(hy) + hy)
        return y, z

    def deep_recursion(self, x, y, z):
        """Run the fixed single pass and return detached recurrent state."""
        y, z = self.latent_recursion(x, y, z)

        # Heads on y
        logits = self.output_head(y)  # [B,T,V]

        # q-head (sequence-level correctness/confidence proxy)
        y_ct = y.transpose(1, 2)  # [B,D,T]
        pooled = self.q_pool(y_ct).squeeze(-1)  # [B,D]
        q_logit = self.q_mlp(pooled).squeeze(-1)  # [B]
        q_hat = torch.sigmoid(q_logit)

        # detach recursive states (paper style)
        return (y.detach(), z.detach()), logits, q_hat


# ----------------------------
# Example wiring
# ----------------------------
if __name__ == "__main__":
    vocab_size = 4096
    d_model = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyRecursiveModel(
        d_model=d_model,
        vocab_size=vocab_size,
        n_heads=8,
        n_layers=4,  # <— 4× Transformer blocks
        dropout=0.1,
        causal=False,  # set True for streamable/causal
        share_nets=False,  # set True to share y/z nets
    ).to(device)

    input_emb = InputEmbedder(vocab_size=vocab_size, d_model=d_model, max_len=8192).to(
        device
    )
