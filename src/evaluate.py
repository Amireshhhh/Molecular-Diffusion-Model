"""
evaluate.py — molecular generation metrics, the ones the field actually reports.

Given generated molecules (coords + atom types), we reconstruct bonds from interatomic
distances and covalent radii (the standard EDM/GEOM approach — no bond info is generated,
so it's inferred from geometry), then score:

  validity   : fraction of generated molecules RDKit can parse into a valid molecule
  uniqueness : fraction of valid molecules that are distinct (by canonical SMILES)
  novelty    : fraction of valid molecules NOT in the training set

These are exactly the metrics EDM (Hoogeboom 2022) and successors report on QM9.
Bond inference from distance is a known-lossy step; we use a standard covalent-radius
lookup with a tolerance, which is the community-standard heuristic.
"""
import numpy as np
from src.data import ATOM_TYPES

# covalent radii in Angstrom (single-bond), for the QM9 atom set
COVALENT_RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57}
BOND_TOL = 0.4     # Angstrom tolerance for calling a bond


def _infer_bonds(coords, sym):
    """Infer bonds from pairwise distances vs. summed covalent radii + tolerance.
    Returns list of (i, j) bonded pairs."""
    n = len(sym)
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            r = COVALENT_RADII[sym[i]] + COVALENT_RADII[sym[j]]
            if d < r + BOND_TOL:
                bonds.append((i, j))
    return bonds


def to_rdkit_mol(coords, type_idx):
    """Build an RDKit mol from coords + atom-type indices, inferring bonds from geometry.
    Returns a sanitized mol or None if invalid."""
    from rdkit import Chem
    from rdkit.Chem import RWMol
    from rdkit.Geometry import Point3D
    sym = [ATOM_TYPES[t] for t in type_idx]
    rw = RWMol()
    for s in sym:
        rw.AddAtom(Chem.Atom(s))
    for i, j in _infer_bonds(coords, sym):
        rw.AddBond(i, j, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    conf = Chem.Conformer(len(sym))
    for i, c in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(c[0]), float(c[1]), float(c[2])))
    mol.AddConformer(conf)
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def evaluate(generated, train_smiles=None):
    """generated: list of (coords (N,3), type_idx (N,)). Returns metric dict."""
    from rdkit import Chem
    valid_smiles = []
    for coords, type_idx in generated:
        mol = to_rdkit_mol(np.asarray(coords), list(type_idx))
        if mol is None:
            continue
        try:
            smi = Chem.MolToSmiles(mol)
            if smi and "." not in smi:      # reject disconnected fragments
                valid_smiles.append(smi)
        except Exception:
            continue
    n = len(generated)
    n_valid = len(valid_smiles)
    uniq = set(valid_smiles)
    metrics = {
        "n_generated": n,
        "validity": n_valid / n if n else 0.0,
        "uniqueness": len(uniq) / n_valid if n_valid else 0.0,
    }
    if train_smiles is not None:
        train_set = set(train_smiles)
        novel = [s for s in uniq if s not in train_set]
        metrics["novelty"] = len(novel) / len(uniq) if uniq else 0.0
    return metrics


def mols_to_smiles(mols):
    """Training molecules (coords,type_idx) -> canonical SMILES set, for novelty."""
    from rdkit import Chem
    out = []
    for coords, type_idx in mols:
        m = to_rdkit_mol(np.asarray(coords), list(type_idx))
        if m is not None:
            try:
                out.append(Chem.MolToSmiles(m))
            except Exception:
                pass
    return out
