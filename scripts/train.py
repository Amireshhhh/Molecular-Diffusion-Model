"""
train.py — train the molecular diffusion model.

Two modes:
  --smoke        : tiny run on synthetic RDKit molecules (CPU, seconds) — proves the loop.
  (default)      : real QM9, sized for a free Colab GPU.

Usage:
  python -m scripts.train --smoke
  python -m scripts.train --data qm9 --epochs 200 --batch 256   # on Colab GPU
"""
import argparse, os, time, torch
from torch.utils.data import DataLoader
from functools import partial
from src.data import make_synthetic, load_qm9, MoleculeDataset, collate, K
from src.model import MoleculeDenoiser
from src.diffusion import Diffusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run on CPU")
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "qm9"])
    ap.add_argument("--qm9_limit", type=int, default=20000, help="cap QM9 size for Colab")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="./ckpt.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.epochs, args.batch, args.T, args.d_model, args.n_layers = 3, 16, 100, 64, 3
        mols = make_synthetic(n=128, seed=0)
    elif args.data == "qm9":
        mols = load_qm9(limit=args.qm9_limit)
    else:
        mols = make_synthetic(n=512, seed=0)
    print(f"[train] device={device} molecules={len(mols)} epochs={args.epochs} batch={args.batch}")

    ds = MoleculeDataset(mols)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=partial(collate))
    in_dim = 3 + K
    model = MoleculeDenoiser(in_dim, d_model=args.d_model, n_layers=args.n_layers).to(device)
    diff = Diffusion(T=args.T, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"[train] params={sum(p.numel() for p in model.parameters()):,}")

    for ep in range(args.epochs):
        model.train(); tot = 0.0; nb = 0; t0 = time.time()
        for x, mask in dl:
            x, mask = x.to(device), mask.to(device)
            opt.zero_grad()
            loss = diff.loss(model, x, mask)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % max(1, args.epochs // 10) == 0 or ep == args.epochs - 1:
            print(f"[train] epoch {ep:4d}  loss {tot/nb:.4f}  ({time.time()-t0:.1f}s)")

    torch.save({"model": model.state_dict(), "args": vars(args), "in_dim": in_dim}, args.out)
    print(f"[train] saved -> {args.out}")


if __name__ == "__main__":
    main()
