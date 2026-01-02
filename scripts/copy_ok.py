from pathlib import Path
import pandas as pd
import shutil

# ====== 需要你改的参数 ======
CSV_PATH = Path("cooper_beta_results.csv")  # 你的csv路径
SRC_DIR  = Path("data")                 # 源文件夹：包含这些文件
DST_DIR  = Path("data-ok")                 # 目标文件夹：要保存到这里

RECURSIVE_SEARCH_IF_MISSING = True   # 直接拼路径找不到时，是否在SRC_DIR下递归按“文件名”搜索
MOVE_INSTEAD_OF_COPY = False         # True=移动；False=复制
OVERWRITE = True                     # 目标已存在是否覆盖
# ==========================


def find_in_src(src_dir: Path, rel_or_name: str, recursive: bool) -> Path | None:
    p = Path(rel_or_name)

    # 1) 按“相对路径/文件名”直接拼到src_dir
    direct = src_dir / p
    if direct.exists():
        return direct

    # 2) 如果csv里带了路径但源目录结构不同，退化为按basename找
    direct2 = src_dir / p.name
    if direct2.exists():
        return direct2

    # 3) 递归搜索（按basename）
    if recursive:
        hits = list(src_dir.rglob(p.name))
        if hits:
            return hits[0]  # 若有多个同名文件，这里取第一个；需要可自行改策略
    return None


def main():
    df = pd.read_csv(CSV_PATH)
    if "filename" not in df.columns:
        raise ValueError("CSV 中找不到列：filename")

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

        dst = DST_DIR / src.name  # 目标里按“文件名”保存（不保留原子目录结构）
        if dst.exists() and not OVERWRITE:
            skipped += 1
            continue

        if MOVE_INSTEAD_OF_COPY:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        copied += 1

    print(f"总计(唯一filename): {len(filenames)}")
    print(f"已处理(复制/移动): {copied}")
    print(f"已跳过(不覆盖): {skipped}")
    print(f"缺失: {len(missing)}")

    if missing:
        miss_path = DST_DIR / "missing_files.txt"
        miss_path.write_text("\n".join(missing), encoding="utf-8")
        print(f"缺失清单已保存: {miss_path}")


if __name__ == "__main__":
    main()
