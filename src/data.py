"""
data.py — molecular data for the diffusion model.

Two sources, identical output format:

  load_qm9(...)         : the REAL QM9 benchmark (~134k small molecules). Downloads on
                          first use. QM9 is THE standard dataset for molecular generation
                          — every paper (EDM, GEOM, etc.) benchmarks on it. Use this on Colab.

  make_synthetic(...)   : a small set of REAL but simple molecules built with RDKit
                          (valid SMILES -> 3D coords via ETKDG). Needs no download, so the
                          pipeline is runnable/testable offline. Use only for smoke tests.

Molecule representation returned by both:
  coords     (N, 3)   3D atom positions, mean-centered
  atom_types (N,)     integer atom-species id (index into ATOM_TYPES)
Collated into padded batches with a node_mask.

QM9 uses 5 atom types: H, C, N, O, F. We keep that vocabulary everywhere.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

ATOM_TYPES = ["H", "C", "N", "O", "F"]        # QM9 vocabulary
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
K = len(ATOM_TYPES)


def _mol_to_arrays(mol):
    """RDKit mol with a 3D conformer -> (coords (N,3), atom_type_idx (N,))."""
    from rdkit import Chem
    conf = mol.GetConformer()
    coords, types = [], []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in ATOM_TO_IDX:
            return None
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x, p.y, p.z])
        types.append(ATOM_TO_IDX[sym])
    c = np.array(coords, dtype=np.float32)
    c -= c.mean(0, keepdims=True)                 # center (translation-invariant target)
    return c, np.array(types, dtype=np.int64)


def make_synthetic(n=256, seed=0):
    """Build n real small molecules from a small SMILES set, embed to 3D with RDKit."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    smiles = ["C", "CC", "CCC", "CCO", "CC=O", "CN", "CCN", "C1CC1", "C=C", "COC",
              "CC(C)C", "CCCC", "CCCO", "C1CCC1", "CC#N", "CCOC", "CC(=O)C", "CO"]
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        smi = smiles[rng.integers(len(smiles))]
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(mol, randomSeed=int(rng.integers(1 << 30))) != 0:
            continue
        arr = _mol_to_arrays(mol)
        if arr is not None and len(arr[1]) <= 29:
            out.append(arr)
    return out


def load_qm9(root="./data/qm9", max_atoms=29, limit=None):
    """Load real QM9 from an .sdf. Downloads the DeepChem QM9 mirror on first use.
    Returns a list of (coords, atom_type_idx). Falls back with a clear error if offline."""
    from rdkit import Chem
    sdf = os.path.join(root, "gdb9.sdf")
    if not os.path.exists(sdf):
        _download_qm9(root)
    supplier = Chem.SDMolSupplier(sdf, removeHs=False, sanitize=True)
    out = []
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        if mol.GetNumConformers() == 0:
            continue
        arr = _mol_to_arrays(mol)
        if arr is None or len(arr[1]) > max_atoms:
            continue
        out.append(arr)
        if limit and len(out) >= limit:
            break
    return out


def _download_qm9(root):
    """Fetch and unpack the QM9 sdf. Runs on a normal network (e.g. Colab)."""
    import urllib.request, tarfile, zipfile, shutil
    os.makedirs(root, exist_ok=True)
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
    zpath = os.path.join(root, "qm9.zip")
    print(f"[data] downloading QM9 from {url} ...")
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    # the zip contains gdb9.sdf (name may vary); normalize
    for name in os.listdir(root):
        if name.endswith(".sdf"):
            shutil.move(os.path.join(root, name), os.path.join(root, "gdb9.sdf"))
            break
    print("[data] QM9 ready.")


class MoleculeDataset(Dataset):
    """Wraps a list of (coords, atom_type_idx) into diffusion-ready tensors.
    Each item: x0 (N, 3+K) = [coords | one-hot types], node_mask handled at collate."""
    def __init__(self, mols, max_atoms=29):
        self.mols = mols
        self.max_atoms = max_atoms

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, i):
        coords, types = self.mols[i]
        n = len(types)
        onehot = np.zeros((n, K), dtype=np.float32)
        onehot[np.arange(n), types] = 1.0
        x0 = np.concatenate([coords, onehot], axis=1)     # (n, 3+K)
        return torch.from_numpy(x0), n


def collate(batch, max_atoms=29):
    """Pad variable-atom molecules to a fixed N, build node_mask."""
    D = 3 + K
    B = len(batch)
    N = max(n for _, n in batch)
    x = torch.zeros(B, N, D)
    mask = torch.zeros(B, N)
    for i, (x0, n) in enumerate(batch):
        x[i, :n] = x0
        mask[i, :n] = 1.0
    return x, mask
