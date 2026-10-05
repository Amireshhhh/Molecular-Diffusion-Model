"""
sample.py — generate molecules from a trained checkpoint and evaluate them.

Usage:
  python -m scripts.sample --ckpt ckpt.pt --n 100
"""
import argparse, torch, numpy as np
from src.model import MoleculeDenoiser
from src.diffusion import Diffusion
from src.data import make_synthetic, K, ATOM_TYPES
from src.evaluate import evaluate, mols_to_smiles


def decode(x, mask):
    """Continuous sample (B,N,3+K) -> list of (coords, atom_type_idx) using the mask."""
    out = []
    for b in range(x.shape[0]):
        n = int(mask[b].sum().item())
        if n == 0:
            continue
        coords = x[b, :n, :3].cpu().numpy()
        types = x[b, :n, 3:].argmax(-1).cpu().numpy()   # argmax the relaxed one-hot back to a type
        out.append((coords, types))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_atoms", type=int, default=15)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.ckpt, map_location=device)
    a = ck["args"]
    model = MoleculeDenoiser(ck["in_dim"], d_model=a["d_model"], n_layers=a["n_layers"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    diff = Diffusion(T=a["T"], device=device)

    # sample atom counts from a simple prior (real pipelines fit this to the data histogram)
    rng = np.random.default_rng(0)
    counts = rng.integers(4, args.max_atoms + 1, size=args.n)
    N = int(counts.max())
    mask = torch.zeros(args.n, N, device=device)
    for i, c in enumerate(counts):
        mask[i, :c] = 1.0

    x = diff.p_sample_loop(model, (args.n, N, 3 + K), mask)
    gen = decode(x, mask)

    train_smiles = mols_to_smiles(make_synthetic(n=128, seed=0))
    m = evaluate(gen, train_smiles=train_smiles)
    print("[sample] metrics:")
    for k, v in m.items():
        print(f"   {k:12s} = {round(v,3) if isinstance(v,float) else v}")


if __name__ == "__main__":
    main()
