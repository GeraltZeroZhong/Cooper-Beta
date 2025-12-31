import os
import shutil
import subprocess
import tempfile
import warnings
import numpy as np
import re
from Bio.PDB import PDBParser, PDBIO
# 精确导入类
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure
from Bio import BiopythonWarning

# 忽略 Biopython 非致命警告
warnings.simplefilter('ignore', BiopythonWarning)

class BarrelExtractor:
    """
    负责从 PDB 文件中提取 β-折叠片段。
    流程：
    1. Sanitize: 清洗 DUM/HOH 原子，修复空链 ID。
    2. Merge: 将所有链合并为单链 A，并强制写入 DSSP 所需的 HEADER。
    3. DSSP: 运行 mkdssp 计算二级结构。
    4. Extract: 映射结果并提取坐标。
    """

    def __init__(self, pdb_file, target_chain_id='A', config=None):
        self.pdb_file = pdb_file
        self.target_chain_id = target_chain_id 
        self.config = config if isinstance(config, dict) else {}

        # --- Expose extra DSSP-derived debug for downstream validators ---
        # 这些字段不会改变 run() 的返回签名，避免破坏现有调用。
        # all_resids: 与 all_coords 对齐的 residue 序号数组 (merged chain A, 1..N)
        # segment_resids: 与 segments 对齐的 residue 序号列表
        # dssp_hbonds: {res_id: [(partner_res_id, energy_kcal_mol), ...]}
        self.all_resids = None
        self.segment_resids = None
        self.dssp_hbonds = None
        # --- Optional AlphaFold pLDDT gate (stored in CA B-factor) ---
        # 仅当检测到数值分布符合 pLDDT 特征时启用，否则不参与判定（避免误伤实验结构）。
        self.all_plddt = None          # np.ndarray aligned with all_coords (0..1) if active
        self.segment_plddt = None      # list of np.ndarray aligned with segments if active
        self.plddt_active = False      # bool
        self.file_exists = False
        try:
            with open(pdb_file, 'r'): pass
            self.file_exists = True
        except:
            self.file_exists = False

    def run(self):
        if not self.file_exists: return None, None

        # 1. 检查 DSSP 工具
        dssp_bin = shutil.which("mkdssp") or shutil.which("dssp") or "/usr/bin/mkdssp"
        if not dssp_bin or not os.path.exists(dssp_bin): 
            print(f"DEBUG: DSSP binary not found! Please install 'dssp' or 'mkdssp'.")
            return None, None
        
        pdb_abspath = os.path.abspath(self.pdb_file)
        
        # 临时文件路径
        sanitized_path = None
        merged_path = None
        output_dssp_path = None

        try:
            # --- Step 1: Sanitize (清洗原始文本) ---
            with tempfile.NamedTemporaryFile(suffix=".pdb", mode='w', delete=False) as tmp_in:
                sanitized_path = tmp_in.name
                try:
                    self._sanitize_opm(pdb_abspath, tmp_in)
                except Exception as e:
                    print(f"DEBUG: Sanitize failed: {e}")
                    return None, None
            
            # --- Step 2: Merge Chains (合并链 + 补充 Header) ---
            with tempfile.NamedTemporaryFile(suffix=".pdb", mode='w', delete=False) as tmp_merged:
                merged_path = tmp_merged.name
            
            if not self._merge_chains_to_A(sanitized_path, merged_path):
                print(f"DEBUG: Merge chains failed for {self.pdb_file}")
                return None, None

            # --- Step 3: Run DSSP ---
            with tempfile.NamedTemporaryFile(suffix=".dssp", delete=False) as tmp_out:
                output_dssp_path = tmp_out.name
            
            cmd = [dssp_bin, merged_path, output_dssp_path]
            
            try:
                # [FIX] 设置超时，快速跳过复杂文件（可在 validator.yaml 中调整）
                timeout_sec = float(self.config.get("dssp_timeout_sec", 5))
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
                
                if result.returncode != 0: 
                    # 只有非超时错误才打印详细日志
                    print(f"DEBUG: DSSP failed (Code {result.returncode}). Stderr: {result.stderr.strip()[:100]}...")
                    return None, None
            
            except subprocess.TimeoutExpired:
                # 捕获超时，不抛出异常，只记录并跳过
                timeout_sec = float(self.config.get("dssp_timeout_sec", 5))
                print(f"DEBUG: DSSP timed out (>{timeout_sec:.1f}s) for {os.path.basename(self.pdb_file)}. Skipping.")
                return None, None
                
            dssp_data = self._parse_dssp_raw(output_dssp_path)
            dssp_hbonds = self._parse_dssp_hbonds(output_dssp_path)

            # --- Step 4: Reload & Map ---
            clean_parser = PDBParser(QUIET=True)
            clean_struct = clean_parser.get_structure('merged', merged_path)
            clean_model = clean_struct[0]
            
            if 'A' not in clean_model: return None, None
            chain_obj = clean_model['A']

                        # --- Step 5: Extract Segments ---
            segments = []
            segment_resids = []
            segment_plddt_raw = []

            current_segment = []
            current_resids = []
            current_plddt_raw = []

            last_resseq = -999
            JUMP_TOL = int(self.config.get("jump_tolerance", 4))
            MIN_SEG_LEN = int(self.config.get("min_segment_len", 3))

            for residue in chain_obj:
                if not residue.has_id('CA'):
                    continue

                r_seq = residue.id[1]
                ss = dssp_data.get(('A', r_seq), '-')

                if ss == 'E':
                    # 若出现较大编号跳跃，切断当前片段
                    if abs(r_seq - last_resseq) > JUMP_TOL:
                        if len(current_segment) >= MIN_SEG_LEN:
                            segments.append(np.array(current_segment))
                            segment_resids.append(np.array(current_resids, dtype=int))
                            segment_plddt_raw.append(np.array(current_plddt_raw, dtype=float))
                        current_segment = []
                        current_resids = []
                        current_plddt_raw = []

                    current_segment.append(residue['CA'].get_coord())
                    current_resids.append(r_seq)
                    # CA B-factor：对 AlphaFold/Predictor 通常为 pLDDT（0..100）
                    current_plddt_raw.append(float(residue['CA'].get_bfactor()))
                    last_resseq = r_seq
                else:
                    if len(current_segment) >= MIN_SEG_LEN:
                        segments.append(np.array(current_segment))
                        segment_resids.append(np.array(current_resids, dtype=int))
                        segment_plddt_raw.append(np.array(current_plddt_raw, dtype=float))
                    current_segment = []
                    current_resids = []
                    current_plddt_raw = []

            if len(current_segment) >= MIN_SEG_LEN:
                segments.append(np.array(current_segment))
                segment_resids.append(np.array(current_resids, dtype=int))
                segment_plddt_raw.append(np.array(current_plddt_raw, dtype=float))

            all_coords = np.concatenate(segments) if segments else np.array([])

            # 对齐后的 residue id 列表，用于拓扑/氢键验证
            all_resids = np.concatenate(segment_resids) if segment_resids else np.array([], dtype=int)

            # --- Optional pLDDT inference & normalization ---
            # 仅当 CA B-factor 分布高度符合 pLDDT（预测结构）时启用；否则不启用门控。
            all_bfactors = np.concatenate(segment_plddt_raw) if segment_plddt_raw else np.array([], dtype=float)
            plddt_active = False
            try:
                det = self.config.get("plddt_detection", {}) if isinstance(self.config, dict) else {}
                det_enable = bool(det.get("enable", True))
                if det_enable and len(all_bfactors) >= 10:
                    bmin_thr = float(det.get("bfactor_min", 0.0))
                    bmax_thr = float(det.get("bfactor_max", 100.0))
                    med_thr = float(det.get("median_min", 50.0))
                    p90_thr = float(det.get("p90_min", 70.0))

                    bmin = float(np.min(all_bfactors))
                    bmax = float(np.max(all_bfactors))
                    if np.isfinite(bmin) and np.isfinite(bmax) and bmin >= bmin_thr and bmax <= bmax_thr:
                        med = float(np.median(all_bfactors))
                        p90 = float(np.percentile(all_bfactors, 90))
                        # Heuristic: predicted structures通常整体偏高；实验 B-factor 通常更低、更分散
                        if med >= med_thr and p90 >= p90_thr:
                            plddt_active = True
            except Exception:
                pass

            self.plddt_active = bool(plddt_active)
            if plddt_active:
                self.segment_plddt = [np.asarray(x, dtype=float) / 100.0 for x in segment_plddt_raw]
                self.all_plddt = np.concatenate(self.segment_plddt) if self.segment_plddt else np.array([], dtype=float)
            else:
                self.segment_plddt = None
                self.all_plddt = None

            self.segment_resids = segment_resids
            self.all_resids = all_resids
            self.dssp_hbonds = dssp_hbonds
            return segments, all_coords

        except Exception as e: 
            # 捕获其他未知错误
            print(f"DEBUG: Global Error parsing {self.pdb_file}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
            
        finally:
            # 清理临时文件
            for path in [sanitized_path, merged_path, output_dssp_path]:
                if path and os.path.exists(path):
                    try: os.remove(path)
                    except: pass

    def _merge_chains_to_A(self, input_pdb, output_pdb):
        """
        读取清洗后的 PDB，将所有链合并为 Chain A。
        强制写入 HEADER 以满足 DSSP 要求。
        """
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure('temp', input_pdb)
            if not structure: return False
            model = structure[0]
        except Exception: return False

        new_chain = Chain('A')
        res_counter = 1
        
        for chain in model:
            for residue in chain:
                residue.id = (residue.id[0], res_counter, ' ')
                new_chain.add(residue)
                res_counter += 1

        new_model = Model(0)
        new_model.add(new_chain)
        new_struct = Structure('merged')
        new_struct.add(new_model)

        io = PDBIO()
        io.set_structure(new_struct)
        
        try:
            with open(output_pdb, 'w') as f:
                # 写入 DSSP 必需的头信息
                f.write("HEADER    MERGED CHAIN A FOR DSSP\n")
                f.write("CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1\n")
                io.save(f)
            return True
        except Exception: return False

    def _sanitize_opm(self, input_path, output_handle):
        """
        仅负责清洗原子数据，不再写入冗余 Header。
        """
        with open(input_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM"): 
                    res_name = line[17:20]
                    if res_name == "DUM" or res_name == "HOH": continue
                    
                    if len(line) > 21 and line[21] == ' ':
                        line = line[:21] + 'A' + line[22:]
                    
                    output_handle.write(line)
                
                elif line.startswith("TER") or line.startswith("END"):
                     output_handle.write(line)
        output_handle.flush()

    def _parse_dssp_raw(self, dssp_file):
        """解析 DSSP 输出"""
        data = {}
        start = False
        with open(dssp_file, 'r') as f:
            for line in f:
                if "  #  RESIDUE" in line:
                    start = True; continue
                if not start: continue
                if len(line) < 20: continue
                try:
                    res_seq = int(line[5:10].strip())
                    ss = line[16]
                    if ss == ' ': ss = '-'
                    data[('A', res_seq)] = ss
                except ValueError: continue
        return data

    def _parse_dssp_hbonds(self, dssp_file):
        """解析 DSSP 输出中的氢键信息。

        说明：mkdssp 的四列氢键 (N-H-->O, O-->H-N, N-H-->O, O-->H-N) 以
        “offset,energy” 的形式给出，其中 offset 为 DSSP 序号偏移（在本项目中
        由于 merge 后连续编号，可近似视作 residue 序号偏移）。

        参考格式示例：-3,-1.4 表示 i 的 HN 与 (i-3) 的 O 成键，能量 -1.4 kcal/mol。
        """
        hb = {}
        start = False
        # 捕获形如 "-3,-1.4" 或 "0, 0.0" 的四组字段
        pat = re.compile(r"(-?\d+),\s*(-?\d+\.\d+)")
        with open(dssp_file, 'r') as f:
            for line in f:
                if "  #  RESIDUE" in line:
                    start = True
                    continue
                if not start:
                    continue
                if len(line) < 20:
                    continue
                try:
                    # 与 _parse_dssp_raw 一致：取合并链的 residue 序号
                    res_seq = int(line[5:10].strip())
                except ValueError:
                    continue

                pairs = pat.findall(line)
                if not pairs:
                    continue

                out = []
                for off_s, e_s in pairs[:4]:
                    try:
                        off = int(off_s)
                        e = float(e_s)
                    except ValueError:
                        continue
                    if off == 0:
                        continue
                    partner = res_seq + off
                    if partner <= 0:
                        continue
                    out.append((partner, e))

                if out:
                    hb[res_seq] = out
        return hb