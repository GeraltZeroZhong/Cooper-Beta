#!/usr/bin/env python3
"""
Download all structures referenced by an mpstruc beta-barrel XML dump and classify
entries at the assembly/entity level for D1-style canonical TMBB curation.

What this script does
---------------------
1. Parse an mpstruc XML dump (master proteins + member proteins + optional related entries).
2. Download entry mmCIF files from RCSB.
3. Download available biological assembly mmCIF files.
4. Parse each entry mmCIF and compute chain/entity-level summary:
   - protein entities
   - chain-to-entity mapping
   - observed residue counts
   - beta-strand counts from _struct_sheet_range
   - preferred assembly composition
5. Produce a preliminary classification useful for D1 curation:
   - SELF_CONTAINED_MONOMER
   - SELF_CONTAINED_HOMOOLIGOMER
   - SELF_CONTAINED_WITH_PARTNER_COMPLEX
   - ASSEMBLY_FORMED_OR_OUT_OF_SCOPE
   - PARTIAL_OR_DOMAIN_ONLY
   - DESIGNED_OR_OUT_OF_SCOPE
   - NEEDS_REVIEW
6. Create symlink/copy based file organization by mpstruc subgroup and by class.

Important note
--------------
The classification is intentionally conservative. It is designed to separate
"good canonical positive candidates" from obvious exclusions and review cases.
You should still inspect the SELF_CONTAINED_WITH_PARTNER_COMPLEX,
PARTIAL_OR_DOMAIN_ONLY, and NEEDS_REVIEW sets manually.

Recommended usage
-----------------
python mpstruc_download_and_classify.py Mpstrucis.txt \
    --out mpstruc_beta_barrels \
    --threads 8 \
    --download-related \
    --link-mode symlink

Dependencies
------------
pip install lxml biopython

Optional override file
----------------------
CSV/TSV with columns:
    pdb_code,class_label,note
Use this to force classification for known special cases.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import gzip
import html
import json
import os
import re
import shutil
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Biopython is required. Install with: pip install biopython\n"
        f"Import error: {e}"
    )

try:
    from lxml import etree  # type: ignore
except Exception:  # pragma: no cover
    etree = None


PDB_ID_RE = re.compile(r"^[0-9A-Za-z]{4}$")
UNKNOWN_STR = ""

# Conservative keyword rules.
OUT_OF_SCOPE_KEYWORDS = [
    "hemolysin",
    "leukocidin",
    "toxin",
    "attack complex",
    "macpf",
    "secretin",
    "tolc",
    "oprm",
    "mtre",
    "cusc",
    "cmec",
    "wza",
    "csgg",
    "letb",
    "pore-forming",
]

PARTIAL_KEYWORDS = [
    "fragment",
    "truncated",
    "isolated beta-barrel domain",
    "isolated beta barrel domain",
    "beta-barrel domain",
    "beta barrel domain",
    "translocator domain",
    "domain only",
    "c-terminal domain",
    "c terminal domain",
]

# Sometimes a known good self-contained barrel appears in complex with a partner.
# You can extend these manually if needed.
MANUAL_CLASS_OVERRIDES_DEFAULT: Dict[str, Tuple[str, str]] = {
    # Example:
    # "8XCJ": ("SELF_CONTAINED_WITH_PARTNER_COMPLEX", "LamB barrel + gpJ partner complex"),
    # "1WP1": ("ASSEMBLY_FORMED_OR_OUT_OF_SCOPE", "OprM-type assembly-formed barrel"),
}


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def normalize_pdb_code(code: str) -> str:
    code = (code or "").strip().upper()
    return code


def sanitize_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", html.unescape(str(x))).strip()


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def safe_get(mapping: Dict[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def listify(mapping: Dict[str, Any], *keys: str) -> List[str]:
    value = safe_get(mapping, *keys)
    if value is None:
        return []
    return as_list(value)


def seq_len_from_cif_string(seq: str) -> int:
    seq = seq or ""
    # Remove CIF quoting artifacts and whitespace/newlines.
    seq = seq.replace("\n", "")
    seq = re.sub(r"\s+", "", seq)
    # Keep letters only.
    seq = re.sub(r"[^A-Za-z]", "", seq)
    return len(seq)


def parse_mpstruc_xml(path: str, include_related: bool = True) -> List[Dict[str, str]]:
    """
    Parse the mpstruc dump using a tolerant line-based state machine.

    Why not a strict XML parser?
    The mpstruc text dump mixes XML-like structure with HTML entities/tags and
    a number of malformed records. For this file, line-based parsing of the
    specific fields we care about is substantially more robust than full XML
    recovery parsing.
    """
    raw_lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()

    tag_re = re.compile(r"<([A-Za-z0-9_:-]+)>(.*?)</\1>")

    def clean_inner_text(s: Optional[str]) -> str:
        s = s or ""
        s = html.unescape(s)
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    records: List[Dict[str, str]] = []

    group_name = ""
    subgroup_name = ""
    current_protein: Optional[Dict[str, Any]] = None
    current_member: Optional[Dict[str, Any]] = None
    current_related_owner: Optional[Dict[str, Any]] = None

    def add_record(
        pdb_code: str,
        source_type: str,
        group_name: str,
        subgroup_name: str,
        name: str,
        species: str,
        taxonomic_domain: str,
        expressed_in_species: str,
        resolution: str,
        description: str,
        master_pdb_code: str,
        parent_pdb_code: str,
    ) -> None:
        code = normalize_pdb_code(pdb_code)
        if not PDB_ID_RE.match(code):
            return
        records.append(
            {
                "pdb_code": code,
                "source_type": source_type,
                "group_name": sanitize_text(group_name),
                "subgroup_name": sanitize_text(subgroup_name),
                "mpstruc_name": sanitize_text(name),
                "species": sanitize_text(species),
                "taxonomic_domain": sanitize_text(taxonomic_domain),
                "expressed_in_species": sanitize_text(expressed_in_species),
                "resolution": sanitize_text(resolution),
                "description": sanitize_text(description),
                "master_pdb_code": normalize_pdb_code(master_pdb_code),
                "parent_pdb_code": normalize_pdb_code(parent_pdb_code),
            }
        )

    def flush_member() -> None:
        nonlocal current_member
        if current_member is None or current_protein is None:
            current_member = None
            return
        master_code = normalize_pdb_code(current_protein.get("pdbCode", ""))
        parent_code = normalize_pdb_code(current_member.get("pdbCode", ""))
        merged = dict(current_protein)
        merged.update({k: v for k, v in current_member.items() if not str(k).startswith("_")})
        add_record(
            pdb_code=parent_code,
            source_type="member",
            group_name=group_name,
            subgroup_name=subgroup_name,
            name=merged.get("name", ""),
            species=merged.get("species", ""),
            taxonomic_domain=merged.get("taxonomicDomain", ""),
            expressed_in_species=merged.get("expressedInSpecies", ""),
            resolution=merged.get("resolution", ""),
            description=merged.get("description", ""),
            master_pdb_code=master_code,
            parent_pdb_code=parent_code,
        )
        if include_related:
            for rel in current_member.get("_related", []):
                add_record(
                    pdb_code=rel,
                    source_type="related_member",
                    group_name=group_name,
                    subgroup_name=subgroup_name,
                    name=merged.get("name", ""),
                    species=merged.get("species", ""),
                    taxonomic_domain=merged.get("taxonomicDomain", ""),
                    expressed_in_species=merged.get("expressedInSpecies", ""),
                    resolution="",
                    description=merged.get("description", ""),
                    master_pdb_code=master_code,
                    parent_pdb_code=parent_code,
                )
        current_member = None

    def flush_protein() -> None:
        nonlocal current_protein
        if current_protein is None:
            return
        # Flush any still-open member first.
        flush_member()
        master_code = normalize_pdb_code(current_protein.get("pdbCode", ""))
        add_record(
            pdb_code=master_code,
            source_type="master",
            group_name=group_name,
            subgroup_name=subgroup_name,
            name=current_protein.get("name", ""),
            species=current_protein.get("species", ""),
            taxonomic_domain=current_protein.get("taxonomicDomain", ""),
            expressed_in_species=current_protein.get("expressedInSpecies", ""),
            resolution=current_protein.get("resolution", ""),
            description=current_protein.get("description", ""),
            master_pdb_code=master_code,
            parent_pdb_code=master_code,
        )
        if include_related:
            for rel in current_protein.get("_related", []):
                add_record(
                    pdb_code=rel,
                    source_type="related_master",
                    group_name=group_name,
                    subgroup_name=subgroup_name,
                    name=current_protein.get("name", ""),
                    species=current_protein.get("species", ""),
                    taxonomic_domain=current_protein.get("taxonomicDomain", ""),
                    expressed_in_species=current_protein.get("expressedInSpecies", ""),
                    resolution="",
                    description=current_protein.get("description", ""),
                    master_pdb_code=master_code,
                    parent_pdb_code=master_code,
                )
        current_protein = None

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue

        # Explicit open/close markers.
        if line == "<subgroup>":
            subgroup_name = ""
            continue
        if line == "</subgroup>":
            subgroup_name = ""
            continue
        if line == "<protein>":
            # Protein tags also appear inside <memberProtein>; only open a new master
            # protein when we are not already inside a member block.
            if current_member is None:
                flush_protein()
                current_protein = {"_related": []}
            continue
        if line == "</protein>":
            if current_member is None:
                flush_protein()
            continue
        if line == "<memberProtein>":
            if current_protein is not None:
                flush_member()
                current_member = {"_related": []}
            continue
        if line == "</memberProtein>":
            flush_member()
            continue
        if line == "<relatedPdbEntries>":
            current_related_owner = current_member if current_member is not None else current_protein
            if current_related_owner is not None and "_related" not in current_related_owner:
                current_related_owner["_related"] = []
            continue
        if line == "</relatedPdbEntries>":
            current_related_owner = None
            continue

        m = tag_re.fullmatch(line)
        if not m:
            continue

        tag, inner = m.group(1), m.group(2)
        inner_text = clean_inner_text(inner)

        if current_related_owner is not None and tag == "pdbCode":
            current_related_owner.setdefault("_related", []).append(inner_text)
            continue

        if current_member is not None:
            current_member[tag] = inner_text
            continue
        if current_protein is not None:
            current_protein[tag] = inner_text
            continue

        if tag == "name":
            if inner_text == "TRANSMEMBRANE PROTEINS: BETA-BARREL":
                group_name = inner_text
            elif inner_text:
                subgroup_name = inner_text

    # Flush trailing open records.
    flush_protein()

    return records


def choose_representative_record(records_for_code: List[Dict[str, str]]) -> Dict[str, str]:
    priority = {"master": 0, "member": 1, "related_master": 2, "related_member": 3}
    rec = sorted(records_for_code, key=lambda r: (priority.get(r["source_type"], 99), r["master_pdb_code"], r["parent_pdb_code"]))[0].copy()
    rec["all_source_types"] = ";".join(sorted({r["source_type"] for r in records_for_code}))
    rec["all_subgroups"] = ";".join(sorted({r["subgroup_name"] for r in records_for_code if r["subgroup_name"]}))
    rec["all_master_pdb_codes"] = ";".join(sorted({r["master_pdb_code"] for r in records_for_code if r["master_pdb_code"]}))
    rec["source_record_count"] = str(len(records_for_code))
    return rec


def load_overrides(path: Optional[str]) -> Dict[str, Tuple[str, str]]:
    overrides = dict(MANUAL_CLASS_OVERRIDES_DEFAULT)
    if not path:
        return overrides
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Override file not found: {path}")
    with p.open("r", encoding="utf-8", newline="") as fh:
        # Support CSV or TSV.
        header_line = fh.readline()
        fh.seek(0)
        dialect = csv.excel_tab if "\t" in header_line else csv.excel
        reader = csv.DictReader(fh, dialect=dialect)
        for row in reader:
            code = normalize_pdb_code(row.get("pdb_code", ""))
            label = sanitize_text(row.get("class_label", ""))
            note = sanitize_text(row.get("note", ""))
            if code and label:
                overrides[code] = (label, note)
    return overrides


def urlopen_with_retries(url: str, timeout: int = 30, retries: int = 2, backoff: float = 1.0):
    last_err: Optional[Exception] = None
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "mpstruc-download-classify/1.0",
                    "Accept": "*/*",
                },
                method="GET",
            )
            return urllib.request.urlopen(req, timeout=timeout)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if i < retries:
                time.sleep(backoff * (2 ** i))
            else:
                raise
    raise RuntimeError(f"Unreachable retry code for URL {url}: {last_err}")


def looks_like_cif(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(256).decode("utf-8", errors="ignore").lstrip()
        return head.startswith("data_") or "_entry.id" in head or "loop_" in head
    except Exception:
        return False


def download_file_variants(urls: Sequence[Tuple[str, str]], out_path: str, timeout: int, retries: int, backoff: float) -> Tuple[bool, str]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True, "exists"

    for url, kind in urls:
        try:
            with urlopen_with_retries(url, timeout=timeout, retries=retries, backoff=backoff) as resp:
                status = getattr(resp, "status", None)
                if status is not None and status != 200:
                    continue
                if kind == "plain":
                    with open(tmp_path, "wb") as out:
                        while True:
                            chunk = resp.read(1024 * 256)
                            if not chunk:
                                break
                            out.write(chunk)
                elif kind == "gz":
                    with gzip.GzipFile(fileobj=resp) as gz, open(tmp_path, "wb") as out:
                        while True:
                            chunk = gz.read(1024 * 256)
                            if not chunk:
                                break
                            out.write(chunk)
                else:
                    raise ValueError(f"Unknown download kind: {kind}")

            if looks_like_cif(tmp_path) and os.path.getsize(tmp_path) > 0:
                os.replace(tmp_path, out_path)
                return True, f"downloaded:{kind}"
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                continue
            return False, f"HTTPError {e.code}: {e.reason}"
        except Exception as e:
            return False, f"Error: {e}"

    return False, "No usable file from provided URLs"


def download_entry_cif(code: str, out_dir: str, timeout: int, retries: int, backoff: float) -> Tuple[str, bool, str, str]:
    code = normalize_pdb_code(code)
    out_path = os.path.join(out_dir, f"{code}.cif")
    urls = [
        (f"https://files.rcsb.org/download/{code}.cif", "plain"),
        (f"https://files.rcsb.org/download/{code}.cif.gz", "gz"),
    ]
    ok, msg = download_file_variants(urls, out_path, timeout, retries, backoff)
    return code, ok, msg, out_path


def download_assembly_cif(code: str, assembly_id: str, out_dir: str, timeout: int, retries: int, backoff: float) -> Tuple[str, str, bool, str, str]:
    code = normalize_pdb_code(code)
    aid = str(assembly_id).strip()
    out_path = os.path.join(out_dir, f"{code}-assembly{aid}.cif")
    urls = [
        (f"https://files.rcsb.org/download/{code}-assembly{aid}.cif", "plain"),
        (f"https://files.rcsb.org/download/{code}-assembly{aid}.cif.gz", "gz"),
    ]
    ok, msg = download_file_variants(urls, out_path, timeout, retries, backoff)
    return code, aid, ok, msg, out_path


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_cif_summary(cif_path: str) -> Dict[str, Any]:
    d = MMCIF2Dict(cif_path)

    entry_id = sanitize_text((as_list(safe_get(d, "_entry.id")) or [Path(cif_path).stem.split("-")[0]])[0]).upper()

    entity_ids = listify(d, "_entity.id")
    entity_types = listify(d, "_entity.type")
    entity_descs = listify(d, "_entity.pdbx_description")

    entity_desc_map: Dict[str, str] = {}
    entity_type_map: Dict[str, str] = {}
    for i, eid in enumerate(entity_ids):
        eid = sanitize_text(eid)
        entity_type_map[eid] = sanitize_text(entity_types[i]) if i < len(entity_types) else ""
        entity_desc_map[eid] = sanitize_text(entity_descs[i]) if i < len(entity_descs) else ""

    entity_poly_eids = listify(d, "_entity_poly.entity_id")
    entity_poly_types = listify(d, "_entity_poly.type")
    entity_poly_seq = listify(d, "_entity_poly.pdbx_seq_one_letter_code_can", "_entity_poly.pdbx_seq_one_letter_code")

    entity_poly_type_map: Dict[str, str] = {}
    entity_seq_len_map: Dict[str, int] = {}
    for i, eid in enumerate(entity_poly_eids):
        eid = sanitize_text(eid)
        entity_poly_type_map[eid] = sanitize_text(entity_poly_types[i]) if i < len(entity_poly_types) else ""
        if i < len(entity_poly_seq):
            entity_seq_len_map[eid] = seq_len_from_cif_string(entity_poly_seq[i])

    # Fallback sequence lengths from entity_poly_seq loop.
    if not entity_seq_len_map:
        eps_eids = listify(d, "_entity_poly_seq.entity_id")
        for eid in eps_eids:
            eid = sanitize_text(eid)
            entity_seq_len_map[eid] = entity_seq_len_map.get(eid, 0) + 1

    struct_asym_ids = listify(d, "_struct_asym.id")
    struct_asym_entity_ids = listify(d, "_struct_asym.entity_id")
    asym_to_entity: Dict[str, str] = {}
    for i, asym in enumerate(struct_asym_ids):
        asym = sanitize_text(asym)
        eid = sanitize_text(struct_asym_entity_ids[i]) if i < len(struct_asym_entity_ids) else ""
        if asym:
            asym_to_entity[asym] = eid

    protein_entity_ids = set()
    for eid, etype in entity_type_map.items():
        poly_type = entity_poly_type_map.get(eid, "").lower()
        if etype.lower() == "polymer" and "polypeptide" in poly_type:
            protein_entity_ids.add(eid)

    # label_asym_id -> auth_asym_id mapping from atom_site
    atom_label_asym = listify(d, "_atom_site.label_asym_id")
    atom_auth_asym = listify(d, "_atom_site.auth_asym_id")
    atom_group_pdb = listify(d, "_atom_site.group_PDB")
    atom_label_seq = listify(d, "_atom_site.label_seq_id")
    atom_comp_id = listify(d, "_atom_site.label_comp_id")

    auth_chain_map: Dict[str, str] = {}
    observed_residues_by_chain: Dict[str, set] = defaultdict(set)

    n_atoms = max(len(atom_label_asym), len(atom_auth_asym), len(atom_group_pdb), len(atom_label_seq), len(atom_comp_id))
    for i in range(n_atoms):
        label_asym = sanitize_text(atom_label_asym[i]) if i < len(atom_label_asym) else ""
        auth_asym = sanitize_text(atom_auth_asym[i]) if i < len(atom_auth_asym) else label_asym
        group_pdb = sanitize_text(atom_group_pdb[i]) if i < len(atom_group_pdb) else ""
        label_seq_id = sanitize_text(atom_label_seq[i]) if i < len(atom_label_seq) else ""
        comp_id = sanitize_text(atom_comp_id[i]) if i < len(atom_comp_id) else ""
        if label_asym and label_asym not in auth_chain_map and auth_asym:
            auth_chain_map[label_asym] = auth_asym
        if not label_asym:
            continue
        if group_pdb not in {"ATOM", "HETATM"}:
            continue
        if label_seq_id in {"", ".", "?"}:
            continue
        if comp_id.upper() == "HOH":
            continue
        observed_residues_by_chain[label_asym].add(label_seq_id)

    observed_len_by_chain = {chain: len(res_ids) for chain, res_ids in observed_residues_by_chain.items()}

    # Count beta-strand ranges per chain.
    ss_beg_chain = listify(d, "_struct_sheet_range.beg_label_asym_id", "_struct_sheet_range.beg_auth_asym_id")
    ss_end_chain = listify(d, "_struct_sheet_range.end_label_asym_id", "_struct_sheet_range.end_auth_asym_id")
    ss_beg_seq = listify(d, "_struct_sheet_range.beg_label_seq_id", "_struct_sheet_range.beg_auth_seq_id")
    ss_end_seq = listify(d, "_struct_sheet_range.end_label_seq_id", "_struct_sheet_range.end_auth_seq_id")
    ss_sheet_id = listify(d, "_struct_sheet_range.sheet_id")
    beta_ranges_by_chain: Dict[str, set] = defaultdict(set)
    n_ss = max(len(ss_beg_chain), len(ss_end_chain), len(ss_beg_seq), len(ss_end_seq), len(ss_sheet_id))
    for i in range(n_ss):
        beg_chain = sanitize_text(ss_beg_chain[i]) if i < len(ss_beg_chain) else ""
        end_chain = sanitize_text(ss_end_chain[i]) if i < len(ss_end_chain) else beg_chain
        if not beg_chain or (end_chain and end_chain != beg_chain):
            continue
        rng = (
            sanitize_text(ss_sheet_id[i]) if i < len(ss_sheet_id) else f"sheet_{i+1}",
            sanitize_text(ss_beg_seq[i]) if i < len(ss_beg_seq) else "",
            sanitize_text(ss_end_seq[i]) if i < len(ss_end_seq) else "",
        )
        beta_ranges_by_chain[beg_chain].add(rng)
    beta_count_by_chain = {chain: len(ranges) for chain, ranges in beta_ranges_by_chain.items()}

    # Assembly composition.
    assembly_ids = unique_preserve_order(listify(d, "_pdbx_struct_assembly.id"))
    assembly_gen_ids = listify(d, "_pdbx_struct_assembly_gen.assembly_id")
    assembly_gen_asym_lists = listify(d, "_pdbx_struct_assembly_gen.asym_id_list")

    assembly_to_chains: Dict[str, List[str]] = defaultdict(list)
    for i, aid in enumerate(assembly_gen_ids):
        aid = sanitize_text(aid)
        asym_list = sanitize_text(assembly_gen_asym_lists[i]) if i < len(assembly_gen_asym_lists) else ""
        chains = []
        for token in asym_list.split(","):
            token = token.strip()
            if token:
                chains.append(token)
        assembly_to_chains[aid].extend(chains)

    preferred_assembly_id = "1" if "1" in assembly_ids or "1" in assembly_to_chains else (assembly_ids[0] if assembly_ids else (sorted(assembly_to_chains.keys())[0] if assembly_to_chains else "1"))
    preferred_assembly_chains = unique_preserve_order(assembly_to_chains.get(preferred_assembly_id, []))
    if not preferred_assembly_chains:
        # Fallback: use all chains in entry.
        preferred_assembly_chains = unique_preserve_order(struct_asym_ids)

    # Build protein entity summary.
    entity_summary: Dict[str, Dict[str, Any]] = {}
    for eid in sorted(protein_entity_ids):
        chains = sorted([asym for asym, ent in asym_to_entity.items() if ent == eid])
        chain_infos = []
        for chain in chains:
            seq_len = entity_seq_len_map.get(eid, 0)
            observed_len = observed_len_by_chain.get(chain, 0)
            beta_n = beta_count_by_chain.get(chain, 0)
            modeled_frac = (observed_len / seq_len) if seq_len else 0.0
            chain_infos.append(
                {
                    "label_asym_id": chain,
                    "auth_asym_id": auth_chain_map.get(chain, chain),
                    "observed_residues": observed_len,
                    "beta_strands": beta_n,
                    "modeled_fraction": modeled_frac,
                }
            )
        entity_summary[eid] = {
            "entity_id": eid,
            "description": entity_desc_map.get(eid, ""),
            "entity_type": entity_type_map.get(eid, ""),
            "poly_type": entity_poly_type_map.get(eid, ""),
            "seq_len": entity_seq_len_map.get(eid, 0),
            "chains": chain_infos,
            "max_beta_strands": max([x["beta_strands"] for x in chain_infos], default=0),
            "max_observed_residues": max([x["observed_residues"] for x in chain_infos], default=0),
            "max_modeled_fraction": max([x["modeled_fraction"] for x in chain_infos], default=0.0),
            "chain_count_entry": len(chains),
            "chain_count_preferred_assembly": sum(1 for ch in preferred_assembly_chains if asym_to_entity.get(ch) == eid),
        }

    protein_entities_in_preferred_assembly = sorted({asym_to_entity.get(ch, "") for ch in preferred_assembly_chains if asym_to_entity.get(ch, "") in protein_entity_ids})

    return {
        "entry_id": entry_id,
        "preferred_assembly_id": preferred_assembly_id,
        "preferred_assembly_chains": preferred_assembly_chains,
        "protein_entity_ids": sorted(protein_entity_ids),
        "protein_entities_in_preferred_assembly": protein_entities_in_preferred_assembly,
        "entity_summary": entity_summary,
        "available_assembly_ids": assembly_ids,
        "asym_to_entity": asym_to_entity,
        "auth_chain_map": auth_chain_map,
        "observed_len_by_chain": observed_len_by_chain,
        "beta_count_by_chain": beta_count_by_chain,
    }


def choose_candidate_entity(entity_summary: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not entity_summary:
        return None
    entities = list(entity_summary.values())
    entities.sort(
        key=lambda x: (
            int(x.get("max_beta_strands", 0)),
            float(x.get("max_modeled_fraction", 0.0)),
            int(x.get("seq_len", 0)),
            int(x.get("max_observed_residues", 0)),
        ),
        reverse=True,
    )
    return entities[0]


def classify_entry(rep: Dict[str, str], cif_summary: Dict[str, Any], overrides: Dict[str, Tuple[str, str]]) -> Dict[str, Any]:
    code = rep["pdb_code"]
    if code in overrides:
        forced_label, forced_note = overrides[code]
        out = {
            "class_label": forced_label,
            "class_reason": f"manual_override: {forced_note}".strip(),
            "canonical_positive_candidate": str(forced_label in {
                "SELF_CONTAINED_MONOMER",
                "SELF_CONTAINED_HOMOOLIGOMER",
                "SELF_CONTAINED_WITH_PARTNER_COMPLEX",
            }),
        }
        return out

    subgroup = rep.get("subgroup_name", "")
    subgroup_l = subgroup.lower()
    combined_text = " ".join(
        [
            rep.get("mpstruc_name", ""),
            rep.get("description", ""),
            subgroup,
        ]
    ).lower()

    candidate = choose_candidate_entity(cif_summary.get("entity_summary", {}))
    if candidate is None:
        return {
            "class_label": "NEEDS_REVIEW",
            "class_reason": "no_protein_entity_detected",
            "canonical_positive_candidate": "False",
        }

    max_beta = int(candidate.get("max_beta_strands", 0))
    modeled_frac = float(candidate.get("max_modeled_fraction", 0.0))
    seq_len = int(candidate.get("seq_len", 0))
    preferred_entity_ids = cif_summary.get("protein_entities_in_preferred_assembly", [])
    n_protein_entities_assembly = len(preferred_entity_ids)
    n_candidate_chains_assembly = int(candidate.get("chain_count_preferred_assembly", 0))

    if "adventitious membrane proteins" in subgroup_l or "attack complexes" in subgroup_l:
        label = "ASSEMBLY_FORMED_OR_OUT_OF_SCOPE"
        reason = "mpstruc_subgroup_adventitious_pore_forming"
    elif "de novo designed" in subgroup_l:
        label = "DESIGNED_OR_OUT_OF_SCOPE"
        reason = "mpstruc_subgroup_de_novo_designed"
    elif any(k in combined_text for k in OUT_OF_SCOPE_KEYWORDS):
        label = "ASSEMBLY_FORMED_OR_OUT_OF_SCOPE"
        reason = "keyword_out_of_scope"
    elif modeled_frac < 0.65 or (seq_len > 0 and max_beta >= 6 and modeled_frac < 0.75 and any(k in combined_text for k in PARTIAL_KEYWORDS)):
        label = "PARTIAL_OR_DOMAIN_ONLY"
        reason = f"low_modeled_fraction_or_partial_keyword(modeled_fraction={modeled_frac:.3f})"
    elif max_beta >= 8:
        if n_protein_entities_assembly <= 1:
            if n_candidate_chains_assembly <= 1:
                label = "SELF_CONTAINED_MONOMER"
                reason = f"candidate_entity_has_{max_beta}_beta_strands_single_chain"
            else:
                label = "SELF_CONTAINED_HOMOOLIGOMER"
                reason = f"candidate_entity_has_{max_beta}_beta_strands_homooligomer"
        else:
            label = "SELF_CONTAINED_WITH_PARTNER_COMPLEX"
            reason = f"candidate_entity_has_{max_beta}_beta_strands_plus_partner_entities"
    elif max_beta <= 5 and n_candidate_chains_assembly >= 3:
        label = "ASSEMBLY_FORMED_OR_OUT_OF_SCOPE"
        reason = f"low_per_chain_beta_strands({max_beta})_multichain_assembly"
    else:
        label = "NEEDS_REVIEW"
        reason = f"ambiguous_beta_strands={max_beta};modeled_fraction={modeled_frac:.3f};protein_entities_in_assembly={n_protein_entities_assembly}"

    return {
        "class_label": label,
        "class_reason": reason,
        "canonical_positive_candidate": str(label in {
            "SELF_CONTAINED_MONOMER",
            "SELF_CONTAINED_HOMOOLIGOMER",
            "SELF_CONTAINED_WITH_PARTNER_COMPLEX",
        }),
    }


def safe_symlink_or_copy(src: str, dst: str, link_mode: str = "symlink") -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return
    if link_mode == "symlink":
        try:
            rel_src = os.path.relpath(src, os.path.dirname(dst))
            os.symlink(rel_src, dst)
            return
        except Exception:
            pass
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        header = fieldnames or sorted(keys)
    else:
        header = fieldnames or []
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and classify mpstruc beta-barrel structures.")
    ap.add_argument("input_xml", help="Path to mpstruc XML dump (e.g. Mpstrucis.txt)")
    ap.add_argument("--out", default="mpstruc_beta_barrels", help="Output directory")
    ap.add_argument("--threads", type=int, default=8, help="Download threads")
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    ap.add_argument("--retries", type=int, default=2, help="Retries per URL")
    ap.add_argument("--backoff", type=float, default=0.8, help="Backoff base seconds")
    ap.add_argument("--download-related", action="store_true", help="Include relatedPdbEntries in addition to master/member entries")
    ap.add_argument("--skip-download", action="store_true", help="Only parse XML and build manifests; do not download files")
    ap.add_argument("--skip-assemblies", action="store_true", help="Download entry files only; skip biological assembly files")
    ap.add_argument("--override-csv", default=None, help="Optional CSV/TSV with pdb_code,class_label,note")
    ap.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink", help="How to organize classified files")
    args = ap.parse_args()

    out_dir = Path(args.out)
    entries_dir = out_dir / "entries"
    assemblies_dir = out_dir / "assemblies"
    meta_dir = out_dir / "metadata"
    by_subgroup_dir = out_dir / "by_subgroup"
    by_class_dir = out_dir / "by_class"
    logs_dir = out_dir / "logs"
    for p in [entries_dir, assemblies_dir, meta_dir, by_subgroup_dir, by_class_dir, logs_dir]:
        p.mkdir(parents=True, exist_ok=True)

    overrides = load_overrides(args.override_csv)

    records = parse_mpstruc_xml(args.input_xml, include_related=args.download_related)
    if not records:
        raise SystemExit("No records parsed from mpstruc XML.")

    write_csv(str(meta_dir / "mpstruc_records.csv"), records)

    records_by_code: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for rec in records:
        records_by_code[rec["pdb_code"]].append(rec)

    unique_entries = [choose_representative_record(records_by_code[code]) for code in sorted(records_by_code.keys())]
    write_csv(str(meta_dir / "mpstruc_unique_entries.csv"), unique_entries)

    print(f"[INFO] Parsed {len(records)} source records, {len(unique_entries)} unique PDB codes.")

    entry_rows: List[Dict[str, Any]] = []
    chain_rows: List[Dict[str, Any]] = []
    download_fail_rows: List[Dict[str, Any]] = []
    assembly_fail_rows: List[Dict[str, Any]] = []

    if args.skip_download:
        print("[INFO] --skip-download enabled. Wrote manifests only.")
        return

    # Step 1: download entry mmCIF files.
    download_results: Dict[str, str] = {}
    with cf.ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
        fut_map = {
            ex.submit(
                download_entry_cif,
                rec["pdb_code"],
                str(entries_dir),
                args.timeout,
                args.retries,
                args.backoff,
            ): rec["pdb_code"]
            for rec in unique_entries
        }
        for fut in cf.as_completed(fut_map):
            code = fut_map[fut]
            try:
                code, ok, msg, out_path = fut.result()
            except Exception as e:
                ok = False
                msg = f"Unhandled exception: {e}"
                out_path = str(entries_dir / f"{code}.cif")
            if ok:
                download_results[code] = out_path
                print(f"[OK] entry {code}: {msg}")
            else:
                download_fail_rows.append({"pdb_code": code, "stage": "entry", "message": msg})
                print(f"[FAIL] entry {code}: {msg}")

    # Step 2: parse entry mmCIF, classify, and download assemblies.
    assembly_jobs: List[Tuple[str, str]] = []
    for rep in unique_entries:
        code = rep["pdb_code"]
        cif_path = download_results.get(code)
        if not cif_path or not os.path.exists(cif_path):
            entry_rows.append(
                {
                    **rep,
                    "class_label": "DOWNLOAD_FAILED",
                    "class_reason": "entry_cif_download_failed",
                    "canonical_positive_candidate": "False",
                }
            )
            continue

        try:
            cif_summary = parse_cif_summary(cif_path)
            class_info = classify_entry(rep, cif_summary, overrides)
            candidate = choose_candidate_entity(cif_summary.get("entity_summary", {}))
            candidate_entity_id = candidate.get("entity_id", "") if candidate else ""
            candidate_label_asym = ""
            candidate_auth_asym = ""
            if candidate and candidate.get("chains"):
                best_chain = sorted(
                    candidate["chains"],
                    key=lambda x: (int(x.get("beta_strands", 0)), float(x.get("modeled_fraction", 0.0)), int(x.get("observed_residues", 0))),
                    reverse=True,
                )[0]
                candidate_label_asym = best_chain.get("label_asym_id", "")
                candidate_auth_asym = best_chain.get("auth_asym_id", "")

            row = {
                **rep,
                **class_info,
                "entry_cif_path": cif_path,
                "preferred_assembly_id": cif_summary.get("preferred_assembly_id", ""),
                "available_assembly_ids": ";".join(cif_summary.get("available_assembly_ids", [])),
                "protein_entity_ids": ";".join(cif_summary.get("protein_entity_ids", [])),
                "protein_entities_in_preferred_assembly": ";".join(cif_summary.get("protein_entities_in_preferred_assembly", [])),
                "candidate_entity_id": candidate_entity_id,
                "candidate_label_asym_id": candidate_label_asym,
                "candidate_auth_asym_id": candidate_auth_asym,
                "candidate_entity_max_beta_strands": candidate.get("max_beta_strands", 0) if candidate else 0,
                "candidate_entity_seq_len": candidate.get("seq_len", 0) if candidate else 0,
                "candidate_entity_max_modeled_fraction": f"{candidate.get('max_modeled_fraction', 0.0):.3f}" if candidate else "0.000",
                "candidate_entity_chain_count_entry": candidate.get("chain_count_entry", 0) if candidate else 0,
                "candidate_entity_chain_count_preferred_assembly": candidate.get("chain_count_preferred_assembly", 0) if candidate else 0,
            }
            entry_rows.append(row)

            for eid, esum in cif_summary.get("entity_summary", {}).items():
                for ch in esum.get("chains", []):
                    chain_rows.append(
                        {
                            "pdb_code": code,
                            "entity_id": eid,
                            "entity_description": esum.get("description", ""),
                            "entity_seq_len": esum.get("seq_len", 0),
                            "label_asym_id": ch.get("label_asym_id", ""),
                            "auth_asym_id": ch.get("auth_asym_id", ""),
                            "observed_residues": ch.get("observed_residues", 0),
                            "beta_strands": ch.get("beta_strands", 0),
                            "modeled_fraction": f"{ch.get('modeled_fraction', 0.0):.3f}",
                            "candidate_entity": str(eid == candidate_entity_id),
                        }
                    )

            if not args.skip_assemblies:
                assembly_ids = cif_summary.get("available_assembly_ids", []) or [cif_summary.get("preferred_assembly_id", "1")]
                for aid in unique_preserve_order([str(x) for x in assembly_ids if str(x).strip()]):
                    assembly_jobs.append((code, aid))

        except Exception as e:
            tb = traceback.format_exc(limit=2)
            entry_rows.append(
                {
                    **rep,
                    "class_label": "PARSE_FAILED",
                    "class_reason": f"parse_error: {e}",
                    "canonical_positive_candidate": "False",
                    "entry_cif_path": cif_path,
                }
            )
            download_fail_rows.append({"pdb_code": code, "stage": "parse", "message": str(e), "traceback": tb})
            print(f"[FAIL] parse {code}: {e}")

    # Step 3: download assembly files.
    if not args.skip_assemblies and assembly_jobs:
        with cf.ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
            fut_map = {
                ex.submit(
                    download_assembly_cif,
                    code,
                    aid,
                    str(assemblies_dir),
                    args.timeout,
                    args.retries,
                    args.backoff,
                ): (code, aid)
                for code, aid in assembly_jobs
            }
            for fut in cf.as_completed(fut_map):
                code, aid = fut_map[fut]
                try:
                    code, aid, ok, msg, out_path = fut.result()
                except Exception as e:
                    ok = False
                    msg = f"Unhandled exception: {e}"
                    out_path = str(assemblies_dir / f"{code}-assembly{aid}.cif")
                if ok:
                    print(f"[OK] assembly {code}-assembly{aid}: {msg}")
                else:
                    assembly_fail_rows.append({"pdb_code": code, "assembly_id": aid, "message": msg})
                    print(f"[FAIL] assembly {code}-assembly{aid}: {msg}")

    # Step 4: write metadata.
    write_csv(str(meta_dir / "entry_classification.csv"), entry_rows)
    write_csv(str(meta_dir / "chain_summary.csv"), chain_rows)
    write_csv(str(logs_dir / "download_failed.csv"), download_fail_rows)
    write_csv(str(logs_dir / "assembly_failed.csv"), assembly_fail_rows)

    positive_candidates = [r for r in entry_rows if r.get("canonical_positive_candidate") == "True"]
    exclusions = [r for r in entry_rows if r.get("class_label") in {"ASSEMBLY_FORMED_OR_OUT_OF_SCOPE", "DESIGNED_OR_OUT_OF_SCOPE"}]
    review_cases = [r for r in entry_rows if r.get("class_label") in {"PARTIAL_OR_DOMAIN_ONLY", "NEEDS_REVIEW", "PARSE_FAILED", "DOWNLOAD_FAILED"}]

    write_csv(str(meta_dir / "d1_positive_candidates.csv"), positive_candidates)
    write_csv(str(meta_dir / "d1_exclusions.csv"), exclusions)
    write_csv(str(meta_dir / "d1_review_cases.csv"), review_cases)

    # Step 5: organize files.
    for row in entry_rows:
        code = row["pdb_code"]
        cif_path = row.get("entry_cif_path", "")
        if not cif_path or not os.path.exists(cif_path):
            continue
        subgroup_slug = slugify(row.get("subgroup_name", ""))
        class_slug = slugify(row.get("class_label", ""))
        subgroup_dst = by_subgroup_dir / subgroup_slug / f"{code}.cif"
        class_dst = by_class_dir / class_slug / f"{code}.cif"
        safe_symlink_or_copy(cif_path, str(subgroup_dst), args.link_mode)
        safe_symlink_or_copy(cif_path, str(class_dst), args.link_mode)

        # Link preferred assembly file if present.
        preferred_aid = str(row.get("preferred_assembly_id", "")).strip()
        if preferred_aid:
            assembly_path = assemblies_dir / f"{code}-assembly{preferred_aid}.cif"
            if assembly_path.exists():
                subgroup_assembly_dst = by_subgroup_dir / subgroup_slug / f"{code}-assembly{preferred_aid}.cif"
                class_assembly_dst = by_class_dir / class_slug / f"{code}-assembly{preferred_aid}.cif"
                safe_symlink_or_copy(str(assembly_path), str(subgroup_assembly_dst), args.link_mode)
                safe_symlink_or_copy(str(assembly_path), str(class_assembly_dst), args.link_mode)

    # Step 6: write a compact JSON summary.
    summary = {
        "n_source_records": len(records),
        "n_unique_pdb_codes": len(unique_entries),
        "n_download_failures": len(download_fail_rows),
        "n_assembly_failures": len(assembly_fail_rows),
        "n_positive_candidates": len(positive_candidates),
        "n_exclusions": len(exclusions),
        "n_review_cases": len(review_cases),
        "class_counts": dict(sorted({k: sum(1 for r in entry_rows if r.get('class_label') == k) for k in sorted({r.get('class_label', '') for r in entry_rows})}.items())),
    }
    with open(meta_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("[DONE] Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[DONE] Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
