# Molecular Diffusion Model (QM9)

A from-scratch **3D denoising-diffusion generative model for small molecules**, trained
on the **QM9** benchmark — the standard dataset for molecular generation. Generates
molecules as 3D atom coordinates + atom types, then evaluates them with the field's
standard metrics (validity, uniqueness, novelty via RDKit).

Built to be understood completely and defended line by line. Every component is verified
by a smoke test that runs on CPU with no download; only full training needs a GPU.

## What it does

1. Represents a molecule as `[3D coords | one-hot atom types]` over the QM9 atom set (H, C, N, O, F).
2. **Diffusion (DDPM, Ho 2020 + cosine schedule):** gradually noises molecules, trains a
   network to reverse it, then generates new molecules from pure noise.
3. **Denoiser:** a permutation-equivariant set-transformer over atoms with sinusoidal
   timestep conditioning.
4. **Evaluation:** infers bonds from geometry (covalent radii), builds RDKit molecules,
   reports validity / uniqueness / novelty — exactly what EDM and successors report on QM9.

## Honest scope — read this

- **Non-equivariant by design.** State-of-the-art molecular diffusion (EDM, Hoogeboom et al.
  2022) uses an **E(3)-equivariant** GNN so that rotating or translating a molecule doesn't
  change its likelihood — a strong inductive bias that improves fidelity. This project uses
  a simpler permutation-equivariant (but not rotation-equivariant) denoiser. That is a
  deliberate, documented simplification for tractability on a free GPU; **the equivariant
  EGNN denoiser is the principled next step**, and it drops into the same diffusion/eval
  scaffold unchanged. Everything else here (the diffusion process, the ε-prediction
  objective, sampling, the metrics) is identical to the real approach.
- **Numbers come from your GPU run.** This repo was verified for *correctness* on CPU
  (the diffusion math, a learning train step, real-molecule loading, and that the metrics
  score real molecules valid and noise invalid). It was **not** trained to convergence
  here — that needs a GPU. Do not quote generation-quality numbers until you've run it.

## Verify it works (no GPU, no download)

```bash
pip install -r requirements.txt
python -m scripts.smoke_test      # 8 checks: diffusion math, train step, data, metrics
python -m scripts.train --smoke   # 3-epoch synthetic run, proves the training loop
```

The smoke run's generated molecules score ~0 validity **on purpose** — a 3-epoch model on
128 toy molecules hasn't learned anything. The same evaluator scores real molecules at
1.0 validity (checked in `smoke_test`), so both the pipeline and the metric are proven.

## Train for real (free Colab GPU)

```bash
python -m scripts.train --data qm9 --qm9_limit 20000 --epochs 200 --batch 256
python -m scripts.sample --ckpt ckpt.pt --n 1000
```

QM9 downloads automatically on first run (a normal network; the sandbox this was built in
blocked it, Colab won't). `--qm9_limit` caps the dataset so it fits free-tier time/memory;
raise it or remove it with more compute. See `notebook.ipynb` for a ready Colab flow.

## Files

- `src/diffusion.py` — the DDPM forward/reverse process and training loss (the math).
- `src/model.py` — the set-transformer denoiser with timestep conditioning.
- `src/data.py` — QM9 loader (+ RDKit synthetic fallback for offline testing).
- `src/evaluate.py` — bond inference + validity/uniqueness/novelty metrics.
- `scripts/train.py`, `scripts/sample.py`, `scripts/smoke_test.py`.

## Why this project (for a GenAI4Science application)

Molecular generative modeling is the exact GenAI4Science domain. This project demonstrates,
in defensible code: the diffusion framework end to end, the 3D-molecule representation, the
QM9 benchmark, and the field's evaluation protocol. The mathematical backbone — a
noising/denoising process over a continuous space — connects directly to optimal-transport
formulations of generative modeling (flow matching, Schrödinger bridges), which is the
natural theory extension to discuss alongside it.

## References

- Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020.
- Nichol & Dhariwal, "Improved DDPM," 2021 (cosine schedule).
- Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D," ICML 2022 (EDM) — the equivariant approach this baseline points toward.
