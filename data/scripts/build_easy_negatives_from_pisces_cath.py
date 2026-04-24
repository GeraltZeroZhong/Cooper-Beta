#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a chain-level easy-negative dataset from PISCES + CATH.

What this script does
---------------------
1. Downloads (or reuses) a precompiled non-redundant PISCES chain list.
2. Downloads (or reuses) the CATH domain list.
3. Maps PISCES chains to CATH class / architecture / topology at chain level.
4. Excludes mpstruc-derived positives and user-provided exclusions.
5. Selects a structurally diverse easy-negative panel (default: 500 chains)
   across CATH classes 1/2/3/4, including Mainly Beta.
6. Downloads the selected full-entry mmCIF files from RCSB.
7. Extracts chain-specific mmCIF files (and PDB files when chain IDs are
   legacy-PDB-compatible) into class-specific folders.
8. Writes complete CSV manifests for candidates, exclusions and final selections.

Notes
-----
- PISCES provides culled PDB chain lists with quality / sequence-identity filters.
- CATH provides domain-level classification; this script converts those domain
  annotations into chain-level dominant class / topology labels.
- The sampling unit here is chain-level, which matches a typical chain-level
  negative set better than directly using CATH domains.
- This script intentionally keeps Mainly Beta in the easy-negative pool if you
  request it (default quotas include class 2).

Typical usage
-------------
python build_easy_negatives_from_pisces_cath.py \
  --out easy_negatives_500 \
  --mpstruc-xml Mpstrucis.txt \
  --n-total 500 \
  --class-quotas 1:125,2:125,3:125,4:125 \
  --pc 20 --resolution-max 2.0 --rmax 0.25 --no-breaks \
  --threads 8

Dependencies
------------
Python >= 3.9
Biopython
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import gzip
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from Bio.PDB import MMCIFIO, MMCIFParser, PDBIO, Select
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Biopython is required. Install it with: pip install biopython"
    ) from e


PISCES_DOWNLOAD_PAGE = "https://dunbrack.fccc.edu/pisces/download/"
PISCES_BASE = "https://dunbrack.fccc.edu/pisces/download/"
CATH_CANDIDATE_URLS = [
    "https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt",
    "https://download.cathdb.info/cath/releases/all-releases/v4_3_0/cath-classification-data/cath-domain-list-v4_3_0.txt",
]
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb}.cif"
RCSB_CIF_GZ_URL = "https://files.rcsb.org/download/{pdb}.cif.gz"

PDB_ID_RE = re.compile(r"^[0-9A-Za-z]{4}$")
DOMAIN_ID_RE = re.compile(r"^[0-9A-Za-z]{7}$")

CLASS_NAME_MAP = {
    1: "Mainly_Alpha",
    2: "Mainly_Beta",
    3: "Alpha_Beta",
    4: "Few_Secondary_Structures",
}


@dataclass
class PiscesRecord:
    pdb_id: str
    chain_raw: str
    method: str
    length: int
    resolution: Optional[float]
    r_value: Optional[float]
    r_free: Optional[float]
    source_line: str


@dataclass
class CathDomain:
    domain_id: str
    pdb_id: str
    chain_id: str
    class_num: int
    architecture: int
    topology: int
    superfamily: int
    domain_len: int
    resolution: Optional[float]


@dataclass
class CathChainSummary:
    pdb_id: str
    chain_id: str
    total_domain_len: int
    domain_count: int
    dominant_class: int
    dominant_class_len: int
    dominant_class_fraction: float
    dominant_architecture: int
    dominant_topology: int
    dominant_superfamily: int
    dominant_cat_len: int
    classes_present: str
    topologies_present: str
    superfamilies_present: str
    domain_ids: str


@dataclass
class CandidateRecord:
    pdb_id: str
    pisces_chain: str
    resolved_chain: str
    map_status: str
    method: str
    chain_length: int
    resolution: Optional[float]
    r_value: Optional[float]
    r_free: Optional[float]
    cath_total_domain_len: int
    cath_domain_count: int
    cath_dominant_class: int
    cath_dominant_class_name: str
    cath_dominant_class_fraction: float
    cath_dominant_architecture: int
    cath_dominant_topology: int
    cath_dominant_superfamily: int
    cath_classes_present: str
    cath_topologies_present: str
    cath_superfamilies_present: str
    cath_domain_ids: str


class ChainOnlySelect(Select):
    def __init__(self, target_chain: str, model_id: int = 0):
        self.target_chain = target_chain
        self.model_id = model_id

    def accept_model(self, model) -> int:
        return 1 if int(model.id) == int(self.model_id) else 0

    def accept_chain(self, chain) -> int:
        return 1 if str(chain.id) == self.target_chain else 0


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: object, path: os.PathLike | str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def http_get(url: str, timeout: int = 60, retries: int = 3, backoff: float = 1.0) -> bytes:
    last_err: Optional[Exception] = None
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "easy-negative-builder/1.0",
                    "Accept": "*/*",
                },
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # pragma: no cover - network dependent
            last_err = e
            if i < retries:
                time.sleep(backoff * (2 ** i))
            else:
                raise
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def download_to_file(url: str, out_path: os.PathLike | str, timeout: int = 60, retries: int = 3, backoff: float = 1.0) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return out
    data = http_get(url, timeout=timeout, retries=retries, backoff=backoff)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    tmp.replace(out)
    return out


def maybe_decompress_gzip_bytes(data: bytes) -> bytes:
    if len(data) >= 2 and data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def normalize_rvalue(x: float) -> str:
    s = f"{x:.2f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


def discover_pisces_filename(
    page_html: str,
    pc: float,
    resolution_max: float,
    rmax: float,
    method: str,
    no_breaks: bool,
    min_len: int,
    max_len: int,
) -> str:
    pc_str = f"{pc:.1f}"
    res_str = f"{resolution_max:.1f}"
    r_str = normalize_rvalue(rmax)
    no_brks = "_noBrks" if no_breaks else ""
    method_pat = re.escape(method)
    pattern = (
        rf"(cullpdb_pc{re.escape(pc_str)}_res0\.0-{re.escape(res_str)}"
        rf"{re.escape(no_brks)}_len{min_len}-{max_len}_R{re.escape(r_str)}_{method_pat}"
        rf"_d\d{{4}}_\d{{2}}_\d{{2}}_chains\d+)"
    )
    matches = re.findall(pattern, page_html)
    if not matches:
        raise RuntimeError(
            "Could not find a matching precompiled PISCES list on the download page. "
            "Try a different combination, or pass --pisces-url manually."
        )
    # Keep the newest one if multiple dated files are present.
    def sort_key(name: str) -> Tuple[str, int]:
        m = re.search(r"_d(\d{4}_\d{2}_\d{2})_chains(\d+)$", name)
        if not m:
            return ("", 0)
        return (m.group(1), int(m.group(2)))

    return sorted(set(matches), key=sort_key, reverse=True)[0]


def get_pisces_url(args: argparse.Namespace, work_dir: Path) -> Tuple[str, Path]:
    if args.pisces_url:
        url = args.pisces_url
        local_name = Path(urllib.parse.urlparse(url).path).name or "pisces_list.txt"
        return url, work_dir / "downloads" / local_name

    html = http_get(PISCES_DOWNLOAD_PAGE, timeout=args.timeout, retries=args.retries, backoff=args.backoff).decode(
        "utf-8", errors="replace"
    )
    fname = discover_pisces_filename(
        page_html=html,
        pc=args.pc,
        resolution_max=args.resolution_max,
        rmax=args.rmax,
        method=args.method,
        no_breaks=args.no_breaks,
        min_len=40,
        max_len=10000,
    )
    return PISCES_BASE + fname, work_dir / "downloads" / fname


def get_cath_file(args: argparse.Namespace, work_dir: Path) -> Path:
    if args.cath_domain_list:
        src = Path(args.cath_domain_list)
        if not src.exists():
            raise FileNotFoundError(f"CATH domain list not found: {src}")
        return src

    out = work_dir / "downloads" / "cath-domain-list.txt"
    if out.exists() and out.stat().st_size > 0:
        return out

    last_err: Optional[Exception] = None
    for url in CATH_CANDIDATE_URLS:
        try:
            data = http_get(url, timeout=args.timeout, retries=args.retries, backoff=args.backoff)
            data = maybe_decompress_gzip_bytes(data)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as f:
                f.write(data)
            return out
        except Exception as e:  # pragma: no cover - network dependent
            last_err = e
            continue
    raise RuntimeError(f"Failed to download CATH domain list: {last_err}")


def parse_float_or_none(s: str) -> Optional[float]:
    s = s.strip()
    if s.upper() in {"NA", "N/A", "NONE", "NULL", "-", ""}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_pisces_list(path: os.PathLike | str) -> List[PiscesRecord]:
    records: List[PiscesRecord] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 7 and PDB_ID_RE.match(parts[0].upper()):
                pdb_id = parts[0].upper()
                chain_raw = parts[1]
                method = parts[2]
                length_s = parts[3]
                resolution_s = parts[4]
                r_value_s = parts[5]
                r_free_s = parts[6]
            elif len(parts) >= 6 and len(parts[0]) >= 5 and PDB_ID_RE.match(parts[0][:4].upper()):
                pdb_id = parts[0][:4].upper()
                chain_raw = parts[0][4:]
                method = parts[2]
                length_s = parts[1]
                resolution_s = parts[3]
                r_value_s = parts[4]
                r_free_s = parts[5]
            else:
                continue
            try:
                length = int(float(length_s))
            except ValueError:
                continue
            rec = PiscesRecord(
                pdb_id=pdb_id,
                chain_raw=chain_raw,
                method=method,
                length=length,
                resolution=parse_float_or_none(resolution_s),
                r_value=parse_float_or_none(r_value_s),
                r_free=parse_float_or_none(r_free_s),
                source_line=line,
            )
            records.append(rec)
    return records


def parse_cath_domain_list(path: os.PathLike | str) -> List[CathDomain]:
    domains: List[CathDomain] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 12:
                continue
            domain_id = parts[0]
            if not DOMAIN_ID_RE.match(domain_id):
                continue
            try:
                class_num = int(parts[1])
                architecture = int(parts[2])
                topology = int(parts[3])
                superfamily = int(parts[4])
                domain_len = int(parts[10])
                resolution = parse_float_or_none(parts[11])
            except ValueError:
                continue
            domains.append(
                CathDomain(
                    domain_id=domain_id,
                    pdb_id=domain_id[:4].upper(),
                    chain_id=domain_id[4],
                    class_num=class_num,
                    architecture=architecture,
                    topology=topology,
                    superfamily=superfamily,
                    domain_len=domain_len,
                    resolution=resolution,
                )
            )
    return domains


def build_cath_chain_summary(domains: Sequence[CathDomain]) -> Tuple[Dict[Tuple[str, str], CathChainSummary], Dict[str, List[str]]]:
    by_chain: Dict[Tuple[str, str], List[CathDomain]] = defaultdict(list)
    entry_to_chains: Dict[str, Set[str]] = defaultdict(set)
    for d in domains:
        by_chain[(d.pdb_id, d.chain_id)].append(d)
        entry_to_chains[d.pdb_id].add(d.chain_id)

    summaries: Dict[Tuple[str, str], CathChainSummary] = {}
    for key, ds in by_chain.items():
        class_len: Dict[int, int] = defaultdict(int)
        cat_len: Dict[Tuple[int, int, int, int], int] = defaultdict(int)
        total_len = 0
        domain_ids: List[str] = []
        classes_present: Set[int] = set()
        topologies_present: Set[str] = set()
        superfamilies_present: Set[str] = set()
        for d in ds:
            total_len += d.domain_len
            class_len[d.class_num] += d.domain_len
            cat_key = (d.class_num, d.architecture, d.topology, d.superfamily)
            cat_len[cat_key] += d.domain_len
            domain_ids.append(d.domain_id)
            classes_present.add(d.class_num)
            topologies_present.add(f"{d.class_num}.{d.architecture}.{d.topology}")
            superfamilies_present.add(f"{d.class_num}.{d.architecture}.{d.topology}.{d.superfamily}")

        dominant_class, dominant_class_len = max(class_len.items(), key=lambda kv: (kv[1], -kv[0]))
        dominant_cat, dominant_cat_len = max(cat_len.items(), key=lambda kv: (kv[1], kv[0]))
        dominant_fraction = dominant_class_len / total_len if total_len > 0 else 0.0
        summaries[key] = CathChainSummary(
            pdb_id=key[0],
            chain_id=key[1],
            total_domain_len=total_len,
            domain_count=len(ds),
            dominant_class=dominant_class,
            dominant_class_len=dominant_class_len,
            dominant_class_fraction=dominant_fraction,
            dominant_architecture=dominant_cat[1],
            dominant_topology=dominant_cat[2],
            dominant_superfamily=dominant_cat[3],
            dominant_cat_len=dominant_cat_len,
            classes_present=";".join(str(x) for x in sorted(classes_present)),
            topologies_present=";".join(sorted(topologies_present)),
            superfamilies_present=";".join(sorted(superfamilies_present)),
            domain_ids=";".join(sorted(domain_ids)),
        )

    entry_to_chains_sorted = {pdb: sorted(chains) for pdb, chains in entry_to_chains.items()}
    return summaries, entry_to_chains_sorted


def extract_pdb_codes_xml_iterparse(path: os.PathLike | str) -> Set[str]:
    codes: Set[str] = set()
    for _event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag.endswith("pdbCode") and elem.text:
            code = elem.text.strip().upper()
            if PDB_ID_RE.match(code):
                codes.add(code)
        elem.clear()
    return codes


def extract_pdb_codes_loose_text(path: os.PathLike | str) -> Set[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return {
        code.upper()
        for code in re.findall(r"<pdbCode>\s*([0-9A-Za-z]{4})\s*</pdbCode>", text)
        if PDB_ID_RE.match(code.upper())
    }


def load_mpstruc_exclusions(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"mpstruc file not found: {p}")
    try:
        return extract_pdb_codes_xml_iterparse(p)
    except ET.ParseError as e:
        codes = extract_pdb_codes_loose_text(p)
        if codes:
            return codes
        raise RuntimeError(f"Failed to parse mpstruc XML: {e}") from e


def load_generic_exclusions(path: Optional[str]) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    """
    Accepts txt/csv/tsv with any of the following:
    - one ID per line: 1ABC or 1ABC_A
    - csv header containing pdb, pdb_id, chain, chain_id
    """
    if not path:
        return set(), set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Exclude file not found: {p}")

    exclude_pdbs: Set[str] = set()
    exclude_chains: Set[Tuple[str, str]] = set()

    def add_token(tok: str) -> None:
        tok = tok.strip().upper()
        if not tok:
            return
        if re.match(r"^[0-9A-Z]{4}$", tok):
            exclude_pdbs.add(tok)
        elif re.match(r"^[0-9A-Z]{4}[_:][^\s]+$", tok):
            pdb, chain = re.split(r"[_:]", tok, maxsplit=1)
            exclude_chains.add((pdb.upper(), chain))

    if p.suffix.lower() in {".csv", ".tsv"}:
        dialect = "excel-tab" if p.suffix.lower() == ".tsv" else "excel"
        with open(p, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, dialect=dialect)
            headers = [h.lower() for h in (reader.fieldnames or [])]
            has_fields = {"pdb", "pdb_id", "chain", "chain_id"} & set(headers)
            if has_fields:
                for row in reader:
                    pdb = (row.get("pdb") or row.get("pdb_id") or "").strip().upper()
                    chain = (row.get("chain") or row.get("chain_id") or "").strip()
                    if pdb and not chain:
                        exclude_pdbs.add(pdb)
                    elif pdb and chain:
                        exclude_chains.add((pdb, chain))
                return exclude_pdbs, exclude_chains

    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            add_token(raw)
    return exclude_pdbs, exclude_chains


def resolve_pisces_to_cath_chain(
    rec: PiscesRecord,
    chain_summaries: Dict[Tuple[str, str], CathChainSummary],
    entry_to_chains: Dict[str, List[str]],
) -> Tuple[Optional[CathChainSummary], str, str]:
    pdb = rec.pdb_id
    raw_chain = rec.chain_raw

    if (pdb, raw_chain) in chain_summaries:
        return chain_summaries[(pdb, raw_chain)], raw_chain, "exact"

    chains = entry_to_chains.get(pdb, [])
    if not chains:
        return None, "", "no_cath_assignment"

    if raw_chain == "0":
        if len(chains) == 1:
            ch = chains[0]
            return chain_summaries[(pdb, ch)], ch, "pisces_zero_unique_chain"

        # If PISCES says "0" but CATH has multiple chains, try the single best length match.
        candidates: List[Tuple[float, str, CathChainSummary]] = []
        for ch in chains:
            sm = chain_summaries[(pdb, ch)]
            if rec.length <= 0:
                diff = abs(sm.total_domain_len)
            else:
                diff = abs(sm.total_domain_len - rec.length) / max(rec.length, 1)
            candidates.append((diff, ch, sm))
        candidates.sort(key=lambda x: (x[0], x[1]))
        if len(candidates) >= 1:
            best = candidates[0]
            second_diff = candidates[1][0] if len(candidates) > 1 else 999.0
            # Accept only if clearly better and reasonably close.
            if best[0] <= 0.25 and (second_diff - best[0] >= 0.15 or len(candidates) == 1):
                return best[2], best[1], "pisces_zero_best_length_match"
        return None, "", "ambiguous_zero_chain"

    if len(chains) == 1:
        ch = chains[0]
        return chain_summaries[(pdb, ch)], ch, "fallback_unique_chain"

    return None, "", "ambiguous_chain"


def make_candidate_rows(
    pisces_records: Sequence[PiscesRecord],
    cath_summaries: Dict[Tuple[str, str], CathChainSummary],
    entry_to_chains: Dict[str, List[str]],
    exclude_pdbs: Set[str],
    exclude_chains: Set[Tuple[str, str]],
    min_length: int,
    max_length: int,
    min_class_fraction: float,
) -> Tuple[List[CandidateRecord], List[Dict[str, str]]]:
    candidates: List[CandidateRecord] = []
    excluded: List[Dict[str, str]] = []

    for rec in pisces_records:
        if rec.pdb_id in exclude_pdbs:
            excluded.append({"pdb_id": rec.pdb_id, "chain": rec.chain_raw, "reason": "excluded_pdb"})
            continue
        if (rec.pdb_id, rec.chain_raw) in exclude_chains:
            excluded.append({"pdb_id": rec.pdb_id, "chain": rec.chain_raw, "reason": "excluded_chain"})
            continue
        if rec.length < min_length or rec.length > max_length:
            excluded.append({
                "pdb_id": rec.pdb_id,
                "chain": rec.chain_raw,
                "reason": f"length_out_of_range:{rec.length}",
            })
            continue

        summary, resolved_chain, map_status = resolve_pisces_to_cath_chain(rec, cath_summaries, entry_to_chains)
        if summary is None:
            excluded.append({"pdb_id": rec.pdb_id, "chain": rec.chain_raw, "reason": map_status})
            continue

        if summary.dominant_class not in {1, 2, 3, 4}:
            excluded.append({
                "pdb_id": rec.pdb_id,
                "chain": resolved_chain,
                "reason": f"unsupported_cath_class:{summary.dominant_class}",
            })
            continue

        if summary.dominant_class_fraction < min_class_fraction:
            excluded.append({
                "pdb_id": rec.pdb_id,
                "chain": resolved_chain,
                "reason": f"dominant_class_fraction_too_low:{summary.dominant_class_fraction:.3f}",
            })
            continue

        candidates.append(
            CandidateRecord(
                pdb_id=rec.pdb_id,
                pisces_chain=rec.chain_raw,
                resolved_chain=resolved_chain,
                map_status=map_status,
                method=rec.method,
                chain_length=rec.length,
                resolution=rec.resolution,
                r_value=rec.r_value,
                r_free=rec.r_free,
                cath_total_domain_len=summary.total_domain_len,
                cath_domain_count=summary.domain_count,
                cath_dominant_class=summary.dominant_class,
                cath_dominant_class_name=CLASS_NAME_MAP.get(summary.dominant_class, f"Class_{summary.dominant_class}"),
                cath_dominant_class_fraction=summary.dominant_class_fraction,
                cath_dominant_architecture=summary.dominant_architecture,
                cath_dominant_topology=summary.dominant_topology,
                cath_dominant_superfamily=summary.dominant_superfamily,
                cath_classes_present=summary.classes_present,
                cath_topologies_present=summary.topologies_present,
                cath_superfamilies_present=summary.superfamilies_present,
                cath_domain_ids=summary.domain_ids,
            )
        )
    return candidates, excluded


def parse_class_quotas(text: str, n_total: int) -> Dict[int, int]:
    if not text:
        base = n_total // 4
        rem = n_total % 4
        quotas = {1: base, 2: base, 3: base, 4: base}
        for cls in [1, 2, 3, 4][:rem]:
            quotas[cls] += 1
        return quotas

    quotas: Dict[int, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        cls_s, n_s = item.split(":", 1)
        cls_i = int(cls_s)
        quotas[cls_i] = int(n_s)
    for cls in [1, 2, 3, 4]:
        quotas.setdefault(cls, 0)
    return quotas


def record_sort_key(rec: CandidateRecord, rng: random.Random) -> Tuple[float, float, float, int, str, str]:
    res = rec.resolution if rec.resolution is not None else 999.0
    rv = rec.r_value if rec.r_value is not None else 999.0
    return (
        res,
        rv,
        -rec.cath_dominant_class_fraction,
        -rec.cath_total_domain_len,
        rec.pdb_id,
        rec.resolved_chain,
    )


def select_diverse_easy_negatives(
    candidates: Sequence[CandidateRecord],
    n_total: int,
    class_quotas: Dict[int, int],
    max_per_topology: int,
    max_per_pdb: int,
    seed: int,
) -> List[CandidateRecord]:
    rng = random.Random(seed)
    pools: Dict[int, List[CandidateRecord]] = defaultdict(list)
    for rec in candidates:
        pools[rec.cath_dominant_class].append(rec)

    for cls in pools:
        rng.shuffle(pools[cls])
        pools[cls].sort(key=lambda r: record_sort_key(r, rng))

    selected: List[CandidateRecord] = []
    selected_keys: Set[Tuple[str, str]] = set()
    per_topology: Counter[str] = Counter()
    per_pdb: Counter[str] = Counter()

    def topo_key(rec: CandidateRecord) -> str:
        return f"{rec.cath_dominant_class}.{rec.cath_dominant_architecture}.{rec.cath_dominant_topology}"

    def try_add(rec: CandidateRecord, enforce_topology_cap: bool, enforce_pdb_cap: bool) -> bool:
        key = (rec.pdb_id, rec.resolved_chain)
        if key in selected_keys:
            return False
        if enforce_pdb_cap and per_pdb[rec.pdb_id] >= max_per_pdb:
            return False
        tk = topo_key(rec)
        if enforce_topology_cap and per_topology[tk] >= max_per_topology:
            return False
        selected.append(rec)
        selected_keys.add(key)
        per_topology[tk] += 1
        per_pdb[rec.pdb_id] += 1
        return True

    # Pass 1: fill class quotas respecting topology and PDB caps.
    for cls in [1, 2, 3, 4]:
        need = class_quotas.get(cls, 0)
        if need <= 0:
            continue
        for rec in pools.get(cls, []):
            if len([x for x in selected if x.cath_dominant_class == cls]) >= need:
                break
            try_add(rec, enforce_topology_cap=True, enforce_pdb_cap=True)

    # Pass 2: fill remaining per-class shortfalls, relax topology cap but keep PDB cap.
    for cls in [1, 2, 3, 4]:
        need = class_quotas.get(cls, 0)
        if need <= 0:
            continue
        current = sum(1 for x in selected if x.cath_dominant_class == cls)
        if current >= need:
            continue
        for rec in pools.get(cls, []):
            if sum(1 for x in selected if x.cath_dominant_class == cls) >= need:
                break
            try_add(rec, enforce_topology_cap=False, enforce_pdb_cap=True)

    # Pass 3: global fill to n_total, respecting topology and PDB caps.
    if len(selected) < n_total:
        all_pool = []
        for cls in [1, 2, 3, 4]:
            all_pool.extend(pools.get(cls, []))
        all_pool.sort(key=lambda r: record_sort_key(r, rng))
        for rec in all_pool:
            if len(selected) >= n_total:
                break
            try_add(rec, enforce_topology_cap=True, enforce_pdb_cap=True)

    # Pass 4: global fill to n_total, relax topology cap but keep PDB cap.
    if len(selected) < n_total:
        all_pool = []
        for cls in [1, 2, 3, 4]:
            all_pool.extend(pools.get(cls, []))
        all_pool.sort(key=lambda r: record_sort_key(r, rng))
        for rec in all_pool:
            if len(selected) >= n_total:
                break
            try_add(rec, enforce_topology_cap=False, enforce_pdb_cap=True)

    return selected[:n_total]


def candidate_to_dict(rec: CandidateRecord) -> Dict[str, object]:
    return asdict(rec)


def write_csv(rows: Sequence[Dict[str, object]], path: os.PathLike | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            pass
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sanitize_chain_for_filename(chain: str) -> str:
    if chain == " ":
        return "BLANK"
    out = re.sub(r"[^0-9A-Za-z._-]", "_", chain)
    return out or "CHAIN"


def download_one_cif(pdb_id: str, out_dir: Path, timeout: int, retries: int, backoff: float) -> Tuple[str, bool, str, Optional[Path]]:
    pdb_id_l = pdb_id.lower()
    out_path = out_dir / f"{pdb_id_l}.cif"
    if out_path.exists() and out_path.stat().st_size > 0:
        return pdb_id, True, "exists", out_path

    urls = [
        (RCSB_CIF_URL.format(pdb=pdb_id_l), False),
        (RCSB_CIF_GZ_URL.format(pdb=pdb_id_l), True),
    ]
    last_msg = ""
    for url, gz in urls:
        try:
            data = http_get(url, timeout=timeout, retries=retries, backoff=backoff)
            if gz:
                data = maybe_decompress_gzip_bytes(data)
            text_head = data[:64].lstrip()
            if not (text_head.startswith(b"data_") or b"_entry.id" in data[:2048]):
                last_msg = f"unexpected_content_from:{url}"
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".cif.tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            tmp.replace(out_path)
            return pdb_id, True, "downloaded", out_path
        except urllib.error.HTTPError as e:  # pragma: no cover - network dependent
            last_msg = f"HTTPError {e.code}"
            continue
        except Exception as e:  # pragma: no cover - network dependent
            last_msg = f"Error {e}"
            continue
    return pdb_id, False, last_msg or "download_failed", None


def extract_chain_files(
    cif_path: Path,
    pdb_id: str,
    chain_id: str,
    out_root: Path,
    class_num: int,
) -> Tuple[bool, str, Optional[Path], Optional[Path]]:
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(cif_path))
    except Exception as e:
        return False, f"parse_failed:{e}", None, None

    model = next(structure.get_models(), None)
    if model is None:
        return False, "no_model", None, None

    chain_ids = [str(ch.id) for ch in model]
    target_chain = chain_id
    if target_chain not in chain_ids:
        if len(chain_ids) == 1:
            target_chain = chain_ids[0]
        else:
            return False, f"chain_not_found:{chain_id};available={';'.join(chain_ids)}", None, None

    class_dir = out_root / "chains" / f"class_{class_num}_{CLASS_NAME_MAP.get(class_num, f'Class_{class_num}') }"
    class_dir.mkdir(parents=True, exist_ok=True)
    chain_tag = sanitize_chain_for_filename(target_chain)
    cif_out = class_dir / f"{pdb_id.lower()}_{chain_tag}.cif"
    pdb_out: Optional[Path] = None

    try:
        cif_io = MMCIFIO()
        cif_io.set_structure(structure)
        cif_io.save(str(cif_out), select=ChainOnlySelect(target_chain, model_id=int(model.id)))
    except Exception as e:
        return False, f"cif_extract_failed:{e}", None, None

    # PDB format only when target chain looks legacy-compatible (single char or blank).
    if len(target_chain) <= 1:
        try:
            pdb_out = class_dir / f"{pdb_id.lower()}_{chain_tag}.pdb"
            pdb_io = PDBIO()
            pdb_io.set_structure(structure)
            pdb_io.save(str(pdb_out), select=ChainOnlySelect(target_chain, model_id=int(model.id)))
        except Exception:
            pdb_out = None

    return True, "ok", cif_out, pdb_out


def run_downloads_and_extraction(
    selected: Sequence[CandidateRecord],
    work_dir: Path,
    threads: int,
    timeout: int,
    retries: int,
    backoff: float,
) -> List[Dict[str, object]]:
    full_dir = ensure_dir(work_dir / "full_entries")
    unique_pdbs = sorted({rec.pdb_id for rec in selected})

    download_results: Dict[str, Tuple[bool, str, Optional[Path]]] = {}
    with cf.ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        future_map = {
            ex.submit(download_one_cif, pdb_id, full_dir, timeout, retries, backoff): pdb_id
            for pdb_id in unique_pdbs
        }
        for fut in cf.as_completed(future_map):
            pdb_id = future_map[fut]
            try:
                pid, ok, msg, path = fut.result()
            except Exception as e:  # pragma: no cover
                pid, ok, msg, path = pdb_id, False, f"download_exception:{e}", None
            download_results[pid] = (ok, msg, path)
            print(f"[DOWNLOAD {'OK' if ok else 'FAIL'}] {pid}: {msg}", file=sys.stderr)

    extraction_rows: List[Dict[str, object]] = []
    for rec in selected:
        ok, dl_msg, cif_path = download_results.get(rec.pdb_id, (False, "not_downloaded", None))
        row = candidate_to_dict(rec)
        row.update({
            "entry_cif_downloaded": ok,
            "entry_cif_message": dl_msg,
            "entry_cif_path": str(cif_path) if cif_path else "",
            "chain_cif_path": "",
            "chain_pdb_path": "",
            "extract_ok": False,
            "extract_message": "",
        })
        if ok and cif_path is not None:
            ex_ok, ex_msg, chain_cif, chain_pdb = extract_chain_files(
                cif_path=cif_path,
                pdb_id=rec.pdb_id,
                chain_id=rec.resolved_chain,
                out_root=work_dir,
                class_num=rec.cath_dominant_class,
            )
            row["extract_ok"] = ex_ok
            row["extract_message"] = ex_msg
            row["chain_cif_path"] = str(chain_cif) if chain_cif else ""
            row["chain_pdb_path"] = str(chain_pdb) if chain_pdb else ""
        extraction_rows.append(row)
    return extraction_rows


def summarize_selection(selected_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_class = Counter()
    by_topology = Counter()
    by_pdb = Counter()
    for row in selected_rows:
        cls = int(row["cath_dominant_class"])
        by_class[cls] += 1
        tk = f"{row['cath_dominant_class']}.{row['cath_dominant_architecture']}.{row['cath_dominant_topology']}"
        by_topology[tk] += 1
        by_pdb[str(row["pdb_id"])] += 1
    return {
        "n_selected": len(selected_rows),
        "class_counts": {str(k): by_class[k] for k in sorted(by_class)},
        "n_unique_pdb": len(by_pdb),
        "n_unique_topologies": len(by_topology),
        "max_selected_per_pdb": max(by_pdb.values()) if by_pdb else 0,
        "max_selected_per_topology": max(by_topology.values()) if by_topology else 0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Build a chain-level easy-negative set from PISCES + CATH, including Mainly Beta."
    )
    ap.add_argument("--out", default="easy_negatives_from_pisces_cath", help="Output directory")

    # PISCES selection
    ap.add_argument("--pisces-url", default=None, help="Direct URL to a PISCES list file. If omitted, auto-discover from PISCES download page.")
    ap.add_argument("--pc", type=float, default=20.0, help="PISCES max sequence identity cutoff, e.g. 20.0")
    ap.add_argument("--resolution-max", type=float, default=2.0, help="PISCES resolution upper bound, e.g. 2.0")
    ap.add_argument("--rmax", type=float, default=0.25, help="PISCES R-factor upper bound, e.g. 0.25")
    ap.add_argument("--method", default="Xray", choices=["Xray", "Xray+EM", "Xray+Nmr+EM"], help="PISCES experiment filter")
    ap.add_argument("--no-breaks", action="store_true", help="Use PISCES lists with noBrks in the filename")

    # Length / classification filtering
    ap.add_argument("--min-length", type=int, default=80, help="Minimum chain length after PISCES parsing")
    ap.add_argument("--max-length", type=int, default=1200, help="Maximum chain length after PISCES parsing")
    ap.add_argument("--min-class-fraction", type=float, default=0.70, help="Minimum dominant CATH class fraction at chain level")

    # Selection
    ap.add_argument("--n-total", type=int, default=500, help="Number of easy negatives to select")
    ap.add_argument(
        "--class-quotas",
        default="1:125,2:125,3:125,4:125",
        help="Comma-separated quotas per CATH class, e.g. 1:125,2:125,3:125,4:125",
    )
    ap.add_argument("--max-per-topology", type=int, default=3, help="Max selected chains per dominant CATH topology")
    ap.add_argument("--max-per-pdb", type=int, default=1, help="Max selected chains per PDB entry")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for within-class tie shuffling")

    # Exclusions
    ap.add_argument("--mpstruc-xml", default=None, help="mpstruc XML/TXT file; all contained PDB IDs are excluded")
    ap.add_argument("--exclude-file", default=None, help="Optional txt/csv/tsv of extra PDBs or PDB_chain IDs to exclude")

    # External files
    ap.add_argument("--cath-domain-list", default=None, help="Local CATH domain list file; if omitted, auto-download")

    # Download / extraction
    ap.add_argument("--threads", type=int, default=8, help="Download threads")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    ap.add_argument("--retries", type=int, default=3, help="HTTP retry count")
    ap.add_argument("--backoff", type=float, default=1.0, help="HTTP retry exponential backoff base")
    ap.add_argument("--no-download-structures", action="store_true", help="Do not download selected structures / extract chain files")
    ap.add_argument("--dry-run", action="store_true", help="Stop after writing candidate / selection manifests")

    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    work_dir = ensure_dir(args.out)
    meta_dir = ensure_dir(work_dir / "metadata")
    downloads_dir = ensure_dir(work_dir / "downloads")

    write_json(vars(args), meta_dir / "run_config.json")

    print("[INFO] Resolving PISCES source...", file=sys.stderr)
    pisces_url, pisces_local = get_pisces_url(args, work_dir)
    download_to_file(pisces_url, pisces_local, timeout=args.timeout, retries=args.retries, backoff=args.backoff)
    print(f"[INFO] PISCES list: {pisces_url}", file=sys.stderr)

    print("[INFO] Resolving CATH domain list...", file=sys.stderr)
    cath_local = get_cath_file(args, work_dir)
    print(f"[INFO] CATH domain list: {cath_local}", file=sys.stderr)

    print("[INFO] Parsing PISCES list...", file=sys.stderr)
    pisces_records = parse_pisces_list(pisces_local)
    print(f"[INFO] Parsed {len(pisces_records)} PISCES chains.", file=sys.stderr)

    print("[INFO] Parsing CATH domain list...", file=sys.stderr)
    cath_domains = parse_cath_domain_list(cath_local)
    cath_summaries, entry_to_chains = build_cath_chain_summary(cath_domains)
    print(f"[INFO] Parsed {len(cath_domains)} CATH domains and {len(cath_summaries)} CATH chains.", file=sys.stderr)

    mpstruc_exclude = load_mpstruc_exclusions(args.mpstruc_xml)
    extra_exclude_pdbs, extra_exclude_chains = load_generic_exclusions(args.exclude_file)
    exclude_pdbs = set(mpstruc_exclude) | set(extra_exclude_pdbs)
    exclude_chains = set(extra_exclude_chains)
    print(
        f"[INFO] Exclusions: {len(exclude_pdbs)} PDBs, {len(exclude_chains)} chain IDs.",
        file=sys.stderr,
    )

    candidates, excluded = make_candidate_rows(
        pisces_records=pisces_records,
        cath_summaries=cath_summaries,
        entry_to_chains=entry_to_chains,
        exclude_pdbs=exclude_pdbs,
        exclude_chains=exclude_chains,
        min_length=args.min_length,
        max_length=args.max_length,
        min_class_fraction=args.min_class_fraction,
    )
    print(f"[INFO] Candidate easy negatives after filtering: {len(candidates)}", file=sys.stderr)
    print(f"[INFO] Excluded rows: {len(excluded)}", file=sys.stderr)

    # Write intermediate metadata.
    write_csv([asdict(x) for x in pisces_records], meta_dir / "pisces_records.csv")
    write_csv([asdict(x) for x in cath_summaries.values()], meta_dir / "cath_chain_summary.csv")
    write_csv([candidate_to_dict(x) for x in candidates], meta_dir / "easy_negative_candidates.csv")
    write_csv(excluded, meta_dir / "excluded_records.csv")

    quotas = parse_class_quotas(args.class_quotas, args.n_total)
    selected = select_diverse_easy_negatives(
        candidates=candidates,
        n_total=args.n_total,
        class_quotas=quotas,
        max_per_topology=args.max_per_topology,
        max_per_pdb=args.max_per_pdb,
        seed=args.seed,
    )
    print(f"[INFO] Selected {len(selected)} easy negatives.", file=sys.stderr)
    write_csv([candidate_to_dict(x) for x in selected], meta_dir / "easy_negative_selected_pre_download.csv")

    if args.dry_run or args.no_download_structures:
        selected_rows = [candidate_to_dict(x) for x in selected]
        summary = summarize_selection(selected_rows)
        write_json(summary, meta_dir / "selection_summary.json")
        write_csv(selected_rows, meta_dir / "easy_negative_selected.csv")
        print(json.dumps(summary, indent=2), file=sys.stderr)
        return

    extraction_rows = run_downloads_and_extraction(
        selected=selected,
        work_dir=work_dir,
        threads=args.threads,
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
    )
    write_csv(extraction_rows, meta_dir / "easy_negative_selected.csv")
    summary = summarize_selection(extraction_rows)
    write_json(summary, meta_dir / "selection_summary.json")
    print(json.dumps(summary, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
