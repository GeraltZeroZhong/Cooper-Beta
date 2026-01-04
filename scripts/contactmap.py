from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from Bio.PDB import MMCIFParser, PDBParser


@dataclass(frozen=True)
class ResidueID:
    resname: str
    resseq: int


def _get_parser(path: Path):
    suf = path.suffix.lower()
    if suf in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True)
    return PDBParser(QUIET=True)


def _pick_atom(res, prefer_cb: bool = True):
    """
    Pick a representative atom for residue:
    - prefer Cβ, but for GLY fall back to Cα
    - if prefer_cb=False, always use Cα (if present)
    """
    if prefer_cb:
        if res.get_resname() == "GLY" and "CA" in res:
            return res["CA"]
        if "CB" in res:
            return res["CB"]
        # fallback: CA if CB missing
        if "CA" in res:
            return res["CA"]
        return None
    else:
        return res["CA"] if "CA" in res else None


def load_chain_coords(
    struct_path: str | os.PathLike,
    chain_id: str = "A",
    model_id: int = 0,
    use_cb: bool = True,
) -> Tuple[np.ndarray, List[ResidueID]]:
    """
    Load representative coordinates (Cβ or Cα) for a given chain.

    Returns
    -------
    coords : (N, 3) float64
    res_ids : list[ResidueID], length N
    """
    path = Path(struct_path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")

    parser = _get_parser(path)
    structure = parser.get_structure(path.stem, str(path))

    try:
        model = structure[model_id]
    except KeyError as e:
        raise ValueError(f"Model {model_id} not found. Available: {[m.id for m in structure]}") from e

    if chain_id not in model:
        raise ValueError(f"Chain {chain_id} not found. Available: {[c.id for c in model]}")

    chain = model[chain_id]

    coords: List[np.ndarray] = []
    res_ids: List[ResidueID] = []

    for res in chain:
        # skip hetero/water etc.
        if res.id[0] != " ":
            continue

        atom = _pick_atom(res, prefer_cb=use_cb)
        if atom is None:
            continue

        # handle disordered atoms: choose selected altloc if present
        try:
            # Bio.PDB may give DisorderedAtom; get_vector/coord works, but keep safe:
            coord = np.asarray(atom.get_coord(), dtype=float)
        except Exception:
            continue

        coords.append(coord)
        res_ids.append(ResidueID(res.get_resname(), int(res.id[1])))

    if not coords:
        raise ValueError("No valid residues/atoms found (after filtering).")

    return np.vstack(coords), res_ids



def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise Euclidean distance matrix.

    Notes
    -----
    O(N^2) memory/time. For very long chains, consider chunking or scipy.spatial.distance.
    """
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def contact_map_from_dist(
    dist: np.ndarray,
    cutoff: float = 8.0,
    min_seq_sep: int = 3,
    drop_diagonal: bool = True,
) -> np.ndarray:
    """
    Build boolean contact map from distance matrix with sequence separation filtering.
    """
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("dist must be a square (N, N) matrix.")

    n = dist.shape[0]
    cm = dist <= float(cutoff)

    # filter near-sequence neighbors
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    cm &= (sep >= int(min_seq_sep))

    if drop_diagonal:
        np.fill_diagonal(cm, False)

    return cm



def _set_pub_rcparams():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def plot_contact_map(
    cm: np.ndarray,
    res_ids: Optional[Sequence[ResidueID]] = None,
    title: Optional[str] = None,
    out: str | os.PathLike = "contact_map.png",
    dpi: int = 600,
    also_save_pdf: bool = True,
    single_column_inches: float = 3.35,  # ~85 mm
    tick_every: int = 50,
) -> Tuple[Path, Optional[Path]]:
    """
    Plot a publication-ready binary contact map.

    Parameters
    ----------
    cm : (N, N) bool or 0/1 array
    res_ids : optional residue identifiers to map ticks to PDB residue numbers
    out : output raster path (png/tif recommended)
    dpi : raster dpi (600 for print)
    also_save_pdf : additionally save vector PDF alongside
    """
    _set_pub_rcparams()

    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be a square (N, N) matrix.")

    n = cm.shape[0]
    out_path = Path(out)
    pdf_path = out_path.with_suffix(".pdf") if also_save_pdf else None

    # Use a square figure for square data; keep it compact
    fig, ax = plt.subplots(figsize=(single_column_inches, single_column_inches), constrained_layout=True)

    ax.imshow(
        cm.astype(np.uint8),
        origin="lower",
        interpolation="none",
        cmap="Greys",   # black=1 (contact), white=0
        vmin=0,
        vmax=1,
        aspect="equal",
    )

    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")
    if title:
        ax.set_title(title, pad=6)

    # ticks: either residue indices (1..N) or PDB residue numbers if provided
    if tick_every and tick_every > 0:
        tick_pos = np.arange(0, n, tick_every)
        if res_ids is not None and len(res_ids) == n:
            tick_lab = [str(res_ids[i].resseq) for i in tick_pos]
        else:
            tick_lab = [str(i + 1) for i in tick_pos]

        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels(tick_lab)
        ax.set_yticklabels(tick_lab)

    # keep spines visible but minimal
    for spine in ax.spines.values():
        spine.set_visible(True)

    # Save raster
    fig.savefig(out_path, dpi=int(dpi))
    # Save vector (recommended for manuscripts)
    if pdf_path is not None:
        fig.savefig(pdf_path)

    plt.close(fig)
    return out_path, pdf_path


def main():
    import argparse

    p = argparse.ArgumentParser(description="Compute and plot a publication-ready protein contact map.")
    p.add_argument("input", help="Structure file (.pdb/.cif/.mmcif)")
    p.add_argument("--chain", default="A", help="Chain ID (default: A)")
    p.add_argument("--model", type=int, default=0, help="Model index (default: 0)")
    p.add_argument("--atom", choices=["cb", "ca"], default="cb", help="Representative atom (default: cb)")
    p.add_argument("--cutoff", type=float, default=8.0, help="Distance cutoff in Å (default: 8.0)")
    p.add_argument("--min-seq-sep", type=int, default=3, help="Minimum sequence separation (default: 3)")
    p.add_argument("--dpi", type=int, default=600, help="Raster DPI (default: 600)")
    p.add_argument("--out", default="contact_map.png", help="Output raster path (default: contact_map.png)")
    p.add_argument("--title", default=None, help="Figure title (optional)")
    p.add_argument("--no-pdf", action="store_true", help="Do not save PDF (vector)")

    args = p.parse_args()

    coords, res_ids = load_chain_coords(
        args.input,
        chain_id=args.chain,
        model_id=args.model,
        use_cb=(args.atom == "cb"),
    )
    dist = distance_matrix(coords)
    cm = contact_map_from_dist(dist, cutoff=args.cutoff, min_seq_sep=args.min_seq_sep)

    plot_contact_map(
        cm,
        res_ids=res_ids,
        title=args.title or f"Chain {args.chain} ({'Cβ' if args.atom == 'cb' else 'Cα'}, {args.cutoff:.1f} Å)",
        out=args.out,
        dpi=args.dpi,
        also_save_pdf=(not args.no_pdf),
    )


if __name__ == "__main__":
    main()
