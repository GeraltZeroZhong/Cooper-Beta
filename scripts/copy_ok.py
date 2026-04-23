from pathlib import Path
import pandas as pd
import shutil

# ====== Configuration you may want to change ======
CSV_PATH = Path("cooper_beta_results.csv")  # Path to the source CSV
SRC_DIR  = Path("data")                     # Source directory that contains the files
DST_DIR  = Path("data-ok")                  # Destination directory

RECURSIVE_SEARCH_IF_MISSING = True   # If direct path lookup fails, search by basename under SRC_DIR
MOVE_INSTEAD_OF_COPY = False         # True = move; False = copy
OVERWRITE = True                     # Whether to overwrite an existing destination file
# ==========================


def find_in_src(src_dir: Path, rel_or_name: str, recursive: bool) -> Path | None:
    p = Path(rel_or_name)

    # 1) Try the relative path or filename directly under src_dir.
    direct = src_dir / p
    if direct.exists():
        return direct

    # 2) If the CSV includes a path but the directory layout changed, try the basename.
    direct2 = src_dir / p.name
    if direct2.exists():
        return direct2

    # 3) Recursive basename search.
    if recursive:
        hits = list(src_dir.rglob(p.name))
        if hits:
            return hits[0]  # If multiple matches exist, keep the first one.
    return None


def main():
    df = pd.read_csv(CSV_PATH)
    if "filename" not in df.columns:
        raise ValueError('CSV is missing the required column: "filename"')

    DST_DIR.mkdir(parents=True, exist_ok=True)

    filenames = (
        df["filename"]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda s: s.ne("")]
        .unique()
        .tolist()
    )

    copied, skipped, missing = 0, 0, []

    for f in filenames:
        src = find_in_src(SRC_DIR, f, RECURSIVE_SEARCH_IF_MISSING)
        if src is None:
            missing.append(f)
            continue

        dst = DST_DIR / src.name  # Save by basename in the destination directory.
        if dst.exists() and not OVERWRITE:
            skipped += 1
            continue

        if MOVE_INSTEAD_OF_COPY:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        copied += 1

    print(f"Total unique filenames: {len(filenames)}")
    print(f"Copied or moved       : {copied}")
    print(f"Skipped (kept existing): {skipped}")
    print(f"Missing               : {len(missing)}")

    if missing:
        miss_path = DST_DIR / "missing_files.txt"
        miss_path.write_text("\n".join(missing), encoding="utf-8")
        print(f"Missing-file list written to: {miss_path}")


if __name__ == "__main__":
    main()
