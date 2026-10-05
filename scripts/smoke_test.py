"""smoke_test.py — verify every component runs, no GPU/download needed. python -m scripts.smoke_test"""
import numpy as np, torch
from src.diffusion import Diffusion
from src.model import MoleculeDenoiser
from src.data import make_synthetic, MoleculeDataset, collate, K
from src.evaluate import evaluate, mols_to_smiles
from functools import partial
from torch.utils.data import DataLoader

def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"); assert cond, name

print("1. diffusion math")
d = Diffusion(T=200)
check("alphas_cumprod decreasing 1->0", d.alphas_cumprod[0] > 0.99 and d.alphas_cumprod[-1] < 0.01)
x0 = torch.randn(4, 9, 3+K); noise = torch.randn_like(x0)
xt0 = d.q_sample(x0, torch.zeros(4, dtype=torch.long), noise)
corr0 = torch.corrcoef(torch.stack([xt0.flatten(), x0.flatten()]))[0, 1]
check("q_sample at t=0 ~= x0 (corr>0.99)", corr0 > 0.99)
xtT = d.q_sample(x0, torch.full((4,), 199, dtype=torch.long), noise)
corrT = torch.corrcoef(torch.stack([xtT.flatten(), noise.flatten()]))[0, 1]
check("q_sample at t=T ~= noise (corr>0.9)", corrT > 0.9)

print("2. model train step")
m = MoleculeDenoiser(3+K, d_model=64, n_layers=2)
mask = torch.ones(4, 9)
l0 = d.loss(m, x0, mask); l0.backward()
check("gradients flow", any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters()))

print("3. real molecules load")
mols = make_synthetic(n=16, seed=0)
check("built real molecules", len(mols) == 16)
dl = DataLoader(MoleculeDataset(mols), batch_size=8, collate_fn=partial(collate))
x, msk = next(iter(dl))
check("one-hot types valid", torch.allclose((x[...,3:]*msk.unsqueeze(-1)).sum(-1), msk))

print("4. evaluation discriminates")
check("real molecules score valid", evaluate(mols)["validity"] > 0.8)
rng = np.random.default_rng(0)
noise_m = [(rng.normal(size=(8,3)).astype('float32')*3, rng.integers(0,5,size=8)) for _ in range(16)]
check("noise scores invalid", evaluate(noise_m)["validity"] < 0.2)

print("\nALL SMOKE TESTS PASSED")
