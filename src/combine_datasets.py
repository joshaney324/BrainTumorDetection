import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

SOURCES = [
    "axial_t1wce_2_class",
    "coronal_t1wce_2_class",
    "sagittal_t1wce_2_class",
]

SPLITS = ["train", "test"]

OUT_DIR = DATA_DIR / "combined"


def copy_files(src_dir: Path, dst_dir: Path, prefix: str) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src_file in src_dir.iterdir():
        dst_file = dst_dir / f"{prefix}_{src_file.name}"
        shutil.copy2(src_file, dst_file)
        count += 1
    return count


def main() -> None:
    print(f"Output directory: {OUT_DIR}")

    totals: dict[str, int] = {"train": 0, "test": 0}

    for source in SOURCES:
        prefix = source.split("_")[0]
        source_dir = DATA_DIR / source

        for split in SPLITS:
            img_src = source_dir / "images" / split
            lbl_src = source_dir / "labels" / split
            img_dst = OUT_DIR / "images" / split
            lbl_dst = OUT_DIR / "labels" / split

            if not img_src.exists():
                print(f"  [skip] {img_src} not found")
                continue

            n_img = copy_files(img_src, img_dst, prefix)
            n_lbl = copy_files(lbl_src, lbl_dst, prefix) if lbl_src.exists() else 0
            totals[split] += n_img
            print(f"  {source}/{split}: {n_img} images, {n_lbl} labels")

    print(f"\nDone. train={totals['train']}  test={totals['test']}")


if __name__ == "__main__":
    main()