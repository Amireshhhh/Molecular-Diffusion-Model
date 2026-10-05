"""
model.py — the denoiser network that predicts noise added to a molecule.

Input:  noised molecule x_t (B, N, D), timestep t (B,), node_mask (B, N)
Output: predicted noise (B, N, D)

Architecture: a set-transformer style network. Atoms are tokens; self-attention lets
each atom's denoising depend on all others (permutation-equivariant, which molecules
require — there's no canonical atom ordering). Timestep is embedded with a sinusoidal
embedding and injected into every layer.

This is permutation-equivariant (good — atoms are a set) but NOT E(3)-equivariant
(rotating the 3D coords changes the output). See README for why that's a deliberate,
documented simplification and what the equivariant upgrade (EGNN) would change.
"""
import math
import torch
import torch.nn as nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding (as in DDPM / Transformers). t: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DenoiserBlock(nn.Module):
    def __init__(self, d_model, n_heads, t_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.SiLU(), nn.Linear(4 * d_model, d_model))
        self.t_proj = nn.Linear(t_dim, d_model)

    def forward(self, h, t_emb, key_padding_mask):
        h = h + self.t_proj(t_emb).unsqueeze(1)                      # inject timestep
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)  # atoms attend to atoms
        h = self.norm1(h + a)
        h = self.norm2(h + self.ff(h))
        return h


class MoleculeDenoiser(nn.Module):
    def __init__(self, in_dim, d_model=128, n_heads=4, n_layers=4, t_dim=128):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.blocks = nn.ModuleList([DenoiserBlock(d_model, n_heads, t_dim) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, in_dim)

    def forward(self, x, t, node_mask):
        # x: (B, N, in_dim); t: (B,); node_mask: (B, N) with 1=real atom
        t_emb = self.t_mlp(timestep_embedding(t, self.t_dim))
        key_padding = (node_mask == 0)                              # True where padding
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, t_emb, key_padding)
        return self.out_proj(h)
