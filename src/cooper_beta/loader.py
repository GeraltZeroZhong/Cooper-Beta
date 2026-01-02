import os
import re
import shutil
import string
import tempfile
import warnings

from Bio import BiopythonWarning
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import is_aa

warnings.simplefilter("ignore", BiopythonWarning)


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
    从 PDB atom name 推断元素符号。返回 Biopython 常见格式：如 'C', 'N', 'O', 'Cl', 'Zn' 等。
    """
    if not atom_name:
        return ""
    s = atom_name.strip()
    if not s:
        return ""

    # 若形如 "1HG1"：去掉前导数字
    if s[0].isdigit() and len(s) >= 2:
        s = s[1:]

    # 提取字母部分
    s = re.sub(r"[^A-Za-z]", "", s)
    if not s:
        return ""
    s = s.upper()

    if len(s) >= 2 and s[:2] in _TWO_LETTER_ELEMENTS:
        return s[0] + s[1].lower()
    return s[0]


def _fill_missing_atom_elements(model) -> int:
    """补全 atom.element 为空或 'X' 的原子；返回修复数量。"""
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
    将 chain.id 为空/空格的链改成合法单字符，避免 mkdssp/gemmi 在 nonpoly 校验中失败。
    返回修复链数量。
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
    某些 PDB 会触发 Biopython parse_pdb_header 的 currentBiomolecule bug。
    即便 get_header=False，某些版本/路径仍可能触发。
    兜底：去掉 REMARK 350 块后重新解析。
    返回临时文件路径（调用方负责删除）。
    """
    fd, out_path = tempfile.mkstemp(suffix=".pdb")
    with open(in_path, "r", errors="ignore") as fin, os.fdopen(fd, "w") as fout:
        for line in fin:
            if line.startswith("REMARK 350"):
                continue
            fout.write(line)
    return out_path


# -------------------------
# DSSP: export protein only
# -------------------------
class _ProteinOnlySelect(Select):
    """只导出氨基酸残基（包括 MSE 等非标准氨基酸）的 ATOM。"""
    def accept_residue(self, residue):
        return 1 if is_aa(residue, standard=False) else 0

    def accept_atom(self, atom):
        return 1


# -------------------------
# Main Loader
# -------------------------
class ProteinLoader:
    """
    加载 PDB/MMCIF，运行 DSSP，按链提取 CA 坐标及是否为 beta-sheet。
    """

    def __init__(self, file_path, model_id=0, dssp_bin=None):
        self.file_path = file_path
        self.model_id = model_id
        self.dssp_bin = dssp_bin

        self.structure = None
        self.model = None
        self.dssp = None

        self._load_structure()

    def _load_structure(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件未找到: {self.file_path}")

        ext = os.path.splitext(self.file_path)[1].lower()

        try:
            if ext in [".cif", ".mmcif"]:
                parser = MMCIFParser(QUIET=True)
                self.structure = parser.get_structure("struct", self.file_path)
            else:
                # 核心：关闭 header 解析
                parser = PDBParser(QUIET=True, PERMISSIVE=True, get_header=False)
                self.structure = parser.get_structure("struct", self.file_path)

            self.model = self.structure[self.model_id]
            return

        except Exception as e:
            # 兜底：只对 PDB 走 REMARK 350 剔除回退
            if ext not in [".cif", ".mmcif"]:
                tmp = None
                try:
                    tmp = _strip_remark_350_to_temp_pdb(self.file_path)
                    parser = PDBParser(QUIET=True, PERMISSIVE=True, get_header=False)
                    self.structure = parser.get_structure("struct", tmp)
                    self.model = self.structure[self.model_id]
                    return
                except Exception as e2:
                    raise ValueError(f"结构解析失败 {self.file_path}: {str(e2)}") from None
                finally:
                    if tmp and os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass

            raise ValueError(f"结构解析失败 {self.file_path}: {str(e)}") from None

    def _run_dssp(self):
        if self.dssp is not None:
            return

        dssp_bin = self.dssp_bin or shutil.which("mkdssp") or shutil.which("dssp")
        if not dssp_bin:
            raise RuntimeError("未找到 mkdssp/dssp，请先安装并加入 PATH。")

        # DSSP 前处理（对 mkdssp/gemmi 严格解析最关键）
        _sanitize_blank_chain_ids(self.model)
        _fill_missing_atom_elements(self.model)

        tmp_path = None
        try:
            # 只导出蛋白 ATOM，剔除 HETATM，避免 nonpoly_scheme strand/duplicate key 问题
            fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
            with os.fdopen(fd, "w") as handle:
                handle.write("HEADER    GENERATED BY LOADER                         \n")
                io = PDBIO()
                io.set_structure(self.model)
                io.save(handle, select=_ProteinOnlySelect())

            self.dssp = DSSP(self.model, tmp_path, dssp=dssp_bin)

        except Exception as e:
            print(f"  [Warn] DSSP 失败 ({os.path.basename(self.file_path)}): {e}")
            self.dssp = {}

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def get_ca_data(self, chain_id):
        if self.dssp is None:
            self._run_dssp()

        chain = self.model[chain_id] if chain_id in self.model else None
        if not chain:
            chains = list(self.model.get_chains())
            if len(chains) == 1:
                chain = chains[0]
            else:
                return []

        data = []
        for res in chain:
            if not is_aa(res, standard=False):
                continue
            if "CA" not in res:
                continue

            dssp_key = (chain.id, res.id)
            ss_code = "-"
            if self.dssp and dssp_key in self.dssp:
                ss_code = self.dssp[dssp_key][2]

            data.append(
                {
                    "res_id": res.id[1],
                    "coord": res["CA"].get_coord(),
                    "is_sheet": ss_code in ("E", "B"),
                }
            )
        return data

    def get_chain_data(self, chain_id):
        return self.get_ca_data(chain_id)
