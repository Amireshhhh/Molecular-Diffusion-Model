"""
diffusion.py — Denoising Diffusion Probabilistic Model (DDPM) core, for 3D molecules.

This is the math of a diffusion model, independent of what it's applied to:
  forward process  q(x_t | x_0): gradually add Gaussian noise over T steps
  reverse process  p(x_{t-1} | x_t): a learned network removes noise step by step

We use the standard DDPM formulation (Ho et al. 2020) with a cosine noise schedule
(Nichol & Dhariwal 2021), which is what molecular diffusion papers build on.

A molecule here is represented as:
  - coords: (N, 3) atom positions in 3D
  - atom_types: (N, K) one-hot over K atom species
We diffuse the CONTINUOUS concatenation [coords | atom_types] jointly. (Atom types
are relaxed to continuous during diffusion and argmax'd back at the end — the standard
trick for applying continuous diffusion to categorical features.)

NOTE ON SCOPE: this denoiser is NOT E(3)-equivariant. State-of-the-art molecular
diffusion (EDM, Hoogeboom 2022) uses an equivariant GNN so that rotating/translating
a molecule doesn't change its likelihood. We deliberately use a simpler permutation-
aware but non-equivariant network here; equivariance is the principled next step and
is discussed in the README. Everything else (the diffusion process, training objective,
sampling) is identical to the real approach.
"""
import math
import torch
import torch.nn as nn


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule for betas (Nichol & Dhariwal 2021). Returns betas of shape (T,)."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class Diffusion:
    """Holds the noise schedule and implements the forward (noising) and reverse (sampling) processes."""

    def __init__(self, T: int = 500, device: str = "cpu"):
        self.T = T
        self.device = device
        betas = cosine_beta_schedule(T).to(device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # precompute quantities used in the closed-form q(x_t | x_0)
        self.sqrt_acp = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: sample x_t from x_0 in CLOSED FORM (no loop needed).
        x_t = sqrt(acp_t) * x0 + sqrt(1 - acp_t) * noise.
        x0: (B, N, D), t: (B,) long, noise: same shape as x0."""
        sqrt_acp = self.sqrt_acp[t].view(-1, 1, 1)
        sqrt_om = self.sqrt_one_minus_acp[t].view(-1, 1, 1)
        return sqrt_acp * x0 + sqrt_om * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, node_mask):
        """Reverse process: start from pure noise, denoise T -> 0. Returns x_0 estimate.
        shape: (B, N, D); node_mask: (B, N) 1 for real atoms, 0 for padding."""
        B = shape[0]
        x = torch.randn(shape, device=self.device) * node_mask.unsqueeze(-1)
        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            eps_theta = model(x, t, node_mask)          # predicted noise
            beta = self.betas[i]
            alpha = self.alphas[i]
            acp = self.alphas_cumprod[i]
            coef = beta / torch.sqrt(1.0 - acp)
            mean = (1.0 / torch.sqrt(alpha)) * (x - coef * eps_theta)
            if i > 0:
                noise = torch.randn(shape, device=self.device)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean
            x = x * node_mask.unsqueeze(-1)             # keep padding atoms at zero
        return x

    def loss(self, model, x0: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Training objective: predict the noise added at a random timestep (simple MSE, Ho 2020).
        This is the standard epsilon-prediction loss."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(x0) * node_mask.unsqueeze(-1)
        x_t = self.q_sample(x0, t, noise) * node_mask.unsqueeze(-1)
        eps_pred = model(x_t, t, node_mask) * node_mask.unsqueeze(-1)
        # MSE only over real atoms
        se = (eps_pred - noise) ** 2
        return se.sum() / (node_mask.sum() * x0.shape[-1] + 1e-8)
