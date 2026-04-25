import gzip
import os
import re
import string
import tempfile
import warnings

from Bio import BiopythonWarning
from Bio.PDB import PDBIO, MMCIFParser, PDBParser, Select
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import is_aa

from .exceptions import ChainNotFoundError, DsspError, InputValidationError, StructureParseError
from .runtime import require_dssp_binary

_DSSP_PDB_MMCIF_WARNING_PATTERN = r".*not seem to be an mmCIF file.*"
_DEFAULT_CRYST1 = (
    "CRYST1 1000.000 1000.000 1000.000  90.00  90.00  90.00 P 1           1          \n"
)


# -------------------------
# Utilities: element / chain
# -------------------------
_TWO_LETTER_ELEMENTS = {
    "CL", "BR", "NA", "MG", "ZN", "FE", "CA", "CU", "NI", "CO", "MN", "SE",
    "SI", "AL", "CD", "HG", "PB", "SR", "CS", "LI", "AG", "AU", "PT", "IR",
    "KR", "XE", "AR", "NE", "HE",
}


def _infer_element_from_atom_name(atom_name: str) -> str:
    """
    Infer an element symbol from a PDB atom name.

    Returns Biopython-style capitalization such as ``C``, ``N``, ``O``, ``Cl``,
    or ``Zn``.
    """
    if not atom_name:
        return ""
    s = atom_name.strip()
    if not s:
        return ""

    # Strip a leading digit from names such as "1HG1".
    if s[0].isdigit() and len(s) >= 2:
        s = s[1:]

    # Keep alphabetic characters only.
    s = re.sub(r"[^A-Za-z]", "", s)
    if not s:
        return ""
    s = s.upper()

    if len(s) >= 2 and s[:2] in _TWO_LETTER_ELEMENTS:
        return s[0] + s[1].lower()
    return s[0]


def _fill_missing_atom_elements(model) -> int:
    """Fill empty or placeholder ``atom.element`` values and return the count."""
    fixed = 0
    for atom in model.get_atoms():
        elem = (getattr(atom, "element", "") or "").strip()
        if elem and elem != "X":
            continue
        inf = _infer_element_from_atom_name(atom.get_name())
        if inf:
            atom.element = inf
            fixed += 1
    return fixed


def _sanitize_blank_chain_ids(model) -> int:
    """
    Replace blank chain IDs with valid single-character IDs.

    This avoids mkdssp/gemmi failures during non-polymer validation.
    Returns the number of chains updated.
    """
    used = set()
    chains = list(model.get_chains())
    for ch in chains:
        cid = (ch.id or "").strip()
        if cid:
            used.add(cid)

    pool = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
    it = (c for c in pool if c not in used)

    fixed = 0
    for ch in chains:
        if (ch.id or "").strip() == "":
            new_id = next(it, "X")
            ch.id = new_id
            used.add(new_id)
            fixed += 1
    return fixed


# -------------------------
# Utilities: PDB header bug workaround
# -------------------------
def _strip_remark_350_to_temp_pdb(in_path: str) -> str:
    """
    Work around a Biopython ``parse_pdb_header`` bug triggered by some PDBs.

    Even with ``get_header=False``, some versions or code paths still hit the
    ``currentBiomolecule`` bug. As a fallback, drop ``REMARK 350`` records and
    parse the structure again.

    Returns the temporary file path; the caller is responsible for deleting it.
    """
    fd, out_path = tempfile.mkstemp(suffix=".pdb")
    with open(in_path, errors="ignore") as fin, os.fdopen(fd, "w") as fout:
        for line in fin:
            if line.startswith("REMARK 350"):
                continue
            fout.write(line)
    return out_path


def _decompress_gzip_to_temp_if_needed(in_path: str) -> str | None:
    with open(in_path, "rb") as handle:
        if handle.read(2) != b"\x1f\x8b":
            return None

    suffix = os.path.splitext(in_path)[1] or ".pdb"
    fd, out_path = tempfile.mkstemp(suffix=suffix)
    with gzip.open(in_path, "rb") as source, os.fdopen(fd, "wb") as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
    return out_path


# -------------------------
# DSSP: export protein only
# -------------------------
class _ProteinOnlySelect(Select):
    """Export only amino-acid residues, including non-standard residues like MSE."""
    def accept_residue(self, residue):
        return 1 if is_aa(residue, standard=False) else 0

    def accept_atom(self, atom):
        return 1


class ProteinLoader:
    """
    Load PDB/mmCIF structures, run DSSP, and extract per-chain CA data.
    """

    def __init__(
        self,
        file_path,
        model_id=0,
        dssp_bin=None,
        fail_on_dssp_error=True,
        strict_chain: bool = True,
    ):
        self.file_path = file_path
        self.model_id = model_id
        self.dssp_bin = dssp_bin
        self.fail_on_dssp_error = bool(fail_on_dssp_error)
        self.strict_chain = bool(strict_chain)

        self.structure = None
        self.model = None
        self.secondary_structure = None
        self.secondary_structure_error = None

        self._load_structure()

    def _load_structure(self):
        if not os.path.exists(self.file_path):
            raise InputValidationError(f"Structure file not found: {self.file_path}")

        input_tmp = _decompress_gzip_to_temp_if_needed(self.file_path)
        parse_path = input_tmp or self.file_path
        ext = os.path.splitext(parse_path)[1].lower()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BiopythonWarning)
                if ext in [".cif", ".mmcif"]:
                    parser = MMCIFParser(QUIET=True)
                    self.structure = parser.get_structure("struct", parse_path)
                else:
                    # Important: disable header parsing.
                    parser = PDBParser(QUIET=True, PERMISSIVE=True, get_header=False)
                    self.structure = parser.get_structure("struct", parse_path)

            self.model = self.structure[self.model_id]
            return

        except Exception as e:
            # Fallback: only PDB files go through the REMARK 350 stripping path.
            if ext not in [".cif", ".mmcif"]:
                tmp = None
                try:
                    tmp = _strip_remark_350_to_temp_pdb(parse_path)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", BiopythonWarning)
                        parser = PDBParser(QUIET=True, PERMISSIVE=True, get_header=False)
                        self.structure = parser.get_structure("struct", tmp)
                    self.model = self.structure[self.model_id]
                    return
                except Exception as e2:
                    raise StructureParseError(
                        f"Failed to parse structure {self.file_path}: {e2}"
                    ) from None
                finally:
                    if tmp and os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass

            raise StructureParseError(f"Failed to parse structure {self.file_path}: {e}") from None
        finally:
            if input_tmp and os.path.exists(input_tmp):
                try:
                    os.remove(input_tmp)
                except OSError:
                    pass

    def _export_protein_only_pdb(self) -> str:
        _sanitize_blank_chain_ids(self.model)
        _fill_missing_atom_elements(self.model)

        fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
        with os.fdopen(fd, "w") as handle:
            handle.write("HEADER    GENERATED BY LOADER                         \n")
            handle.write(_DEFAULT_CRYST1)
            io = PDBIO()
            io.set_structure(self.model)
            io.save(handle, select=_ProteinOnlySelect())
        return tmp_path

    def _run_dssp(self, tmp_path: str) -> dict[tuple[str, tuple[str, int, str]], str]:
        dssp_bin = require_dssp_binary(self.dssp_bin)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=_DSSP_PDB_MMCIF_WARNING_PATTERN,
                category=UserWarning,
            )
            dssp_result = DSSP(self.model, tmp_path, dssp=dssp_bin)
        return {dssp_key: str(dssp_result[dssp_key][2]) for dssp_key in dssp_result.keys()}

    def _run_secondary_structure(self):
        if self.secondary_structure is not None:
            return

        tmp_path = None
        try:
            # Export protein ATOM records only. Dropping HETATM helps avoid
            # nonpoly_scheme strand/duplicate key issues.
            tmp_path = self._export_protein_only_pdb()
            self.secondary_structure = self._run_dssp(tmp_path)

        except Exception as e:
            self.secondary_structure_error = f"DSSP failed for {os.path.basename(self.file_path)}: {e}"
            if self.fail_on_dssp_error:
                raise DsspError(self.secondary_structure_error) from e
            print(f"  [Warning] {self.secondary_structure_error}")
            self.secondary_structure = {}

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def available_chains(self) -> list[str]:
        """Return chain IDs available in the selected model."""
        return [str(chain.id) for chain in self.model.get_chains()]

    def get_ca_data(self, chain_id, *, strict_chain: bool | None = None):
        chain = self.model[chain_id] if chain_id in self.model else None
        if chain is None:
            chains = list(self.model.get_chains())
            strict = self.strict_chain if strict_chain is None else bool(strict_chain)
            if not strict and len(chains) == 1:
                chain = chains[0]
            else:
                available = ", ".join(self.available_chains()) or "none"
                raise ChainNotFoundError(
                    f"Chain {chain_id!r} not found in {os.path.basename(self.file_path)}. "
                    f"Available chains: {available}."
                )

        if self.secondary_structure is None:
            self._run_secondary_structure()

        data = []
        for res in chain:
            if not is_aa(res, standard=False):
                continue
            if "CA" not in res:
                continue

            dssp_key = (chain.id, res.id)
            ss_code = "-"
            if self.secondary_structure and dssp_key in self.secondary_structure:
                ss_code = self.secondary_structure[dssp_key]

            data.append(
                {
                    "res_id": res.id[1],
                    "chain": chain.id,
                    "coord": res["CA"].get_coord(),
                    "is_sheet": ss_code in ("E", "B"),
                }
            )
        return data

    def get_chain_data(self, chain_id, *, strict_chain: bool | None = None):
        return self.get_ca_data(chain_id, strict_chain=strict_chain)
