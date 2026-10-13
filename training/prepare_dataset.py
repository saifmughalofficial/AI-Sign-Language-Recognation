"""Step 1 of 4: sort a downloaded dataset into dataset/processed/.

    python training/prepare_dataset.py                 # prepare everything found in dataset/raw
    python training/prepare_dataset.py --max-per-class 300 --dry-run

Understands two very common layouts (searched recursively):

1. Folder-per-class  (Kaggle "ASL Alphabet", Roboflow "Folder Structure" export)
       raw/anything/A/img1.jpg   raw/anything/B/img2.jpg ...
2. YOLO detection    (Roboflow "YOLO" export with data.yaml + labels/*.txt)
       the hand is cropped out of each picture using its bounding box.

Result:
       dataset/processed/alphabet/A/000001.jpg ...
       dataset/processed/numbers/5/000001.jpg ...
       dataset/processed/words/HELLO/000001.jpg ...

J and Z are skipped on purpose: they need movement (see README > Dynamic gestures).
"""
from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.constants import DYNAMIC_LETTERS, IMAGE_EXTENSIONS, PROCESSED_DIRS  # noqa: E402
from core.paths import PROCESSED_DIR, RAW_DIR  # noqa: E402

IGNORED_CLASSES = {"NOTHING", "NONE", "SPACE", "DEL", "DELETE", "BACKGROUND", "BLANK", "NEGATIVE"}
NUMBER_WORDS = {"ZERO": "0", "ONE": "1", "TWO": "2", "THREE": "3", "FOUR": "4",
                "FIVE": "5", "SIX": "6", "SEVEN": "7", "EIGHT": "8", "NINE": "9"}
SPLIT_FOLDERS = {"train", "test", "valid", "val", "validation", "images", "labels"}


def normalize_label(raw: str) -> str:
    return re.sub(r"[\s_\-]+", " ", raw.strip()).upper()


def classify_label(label: str) -> tuple[str | None, str]:
    """(category, cleaned_label). category None => ignore this class."""
    label = NUMBER_WORDS.get(label, label)
    if label in IGNORED_CLASSES or not label:
        return None, label
    if len(label) == 1 and label.isalpha():
        return "alphabet", label
    if label.isdigit() and len(label) == 1:
        return "number", label
    return "word", label


# --------------------------------------------------------------------- readers
def iter_folder_layout(root: Path):
    """Yield (image_path, label) where the label is the parent folder name."""
    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            parent = path.parent.name
            if parent.lower() in SPLIT_FOLDERS:
                continue       # YOLO-style "images" folder: handled by iter_yolo_layout
            yield path, normalize_label(parent), None


def read_yolo_names(yaml_path: Path) -> list[str]:
    """Tiny parser for the `names:` entry of a Roboflow data.yaml (no PyYAML needed)."""
    text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^names:\s*(\[.*?\])\s*$", text, re.M | re.S)
    if match:
        try:
            return [str(n) for n in ast.literal_eval(match.group(1))]
        except (ValueError, SyntaxError):
            pass
    names: list[str] = []
    in_block = False
    for line in text.splitlines():
        if re.match(r"^names:\s*$", line):
            in_block = True
            continue
        if in_block:
            item = re.match(r"^\s*-\s*(.+?)\s*$", line) or re.match(r"^\s*\d+\s*:\s*(.+?)\s*$", line)
            if item:
                names.append(item.group(1).strip("'\""))
            elif line.strip() and not line.startswith(" "):
                break
    return names


def iter_yolo_layout(root: Path):
    for yaml_path in root.rglob("data.yaml"):
        names = read_yolo_names(yaml_path)
        if not names:
            continue
        base = yaml_path.parent
        for label_file in base.rglob("*.txt"):
            if label_file.parent.name != "labels":
                continue
            image = None
            for ext in IMAGE_EXTENSIONS:
                candidate = label_file.parent.parent / "images" / (label_file.stem + ext)
                if candidate.exists():
                    image = candidate
                    break
            if image is None:
                continue
            for line in label_file.read_text(errors="ignore").splitlines()[:1]:   # first box only
                parts = line.split()
                if len(parts) >= 5 and parts[0].isdigit() and int(parts[0]) < len(names):
                    box = tuple(float(v) for v in parts[1:5])
                    yield image, normalize_label(names[int(parts[0])]), box


def crop_to_box(src: Path, dst: Path, box: tuple[float, float, float, float]) -> bool:
    import cv2
    img = cv2.imread(str(src))
    if img is None:
        return False
    h, w = img.shape[:2]
    cx, cy, bw, bh = box
    pad = 0.25                                  # keep some space around the hand
    x1 = max(int((cx - bw / 2 * (1 + pad)) * w), 0)
    x2 = min(int((cx + bw / 2 * (1 + pad)) * w), w)
    y1 = max(int((cy - bh / 2 * (1 + pad)) * h), 0)
    y2 = min(int((cy + bh / 2 * (1 + pad)) * h), h)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return False
    return bool(cv2.imwrite(str(dst), img[y1:y2, x1:x2]))


# ------------------------------------------------------------------------ main
def prepare(raw_dir: Path, out_dir: Path, max_per_class: int, dry_run: bool) -> dict:
    items = list(iter_folder_layout(raw_dir)) + list(iter_yolo_layout(raw_dir))
    per_class: dict[tuple[str, str], int] = defaultdict(int)
    skipped_dynamic: set[str] = set()
    skipped_ignored: set[str] = set()
    for src, label, box in items:
        category, label = classify_label(label)
        if category is None:
            skipped_ignored.add(label)
            continue
        if category == "alphabet" and label in DYNAMIC_LETTERS:
            skipped_dynamic.add(label)
            continue
        key = (category, label)
        if per_class[key] >= max_per_class:
            continue
        per_class[key] += 1
        if dry_run:
            continue
        folder = out_dir / PROCESSED_DIRS[category] / label
        folder.mkdir(parents=True, exist_ok=True)
        dst = folder / f"{per_class[key]:06d}{src.suffix.lower() if box is None else '.jpg'}"
        if box is not None:
            if not crop_to_box(src, dst, box):
                per_class[key] -= 1
        else:
            shutil.copy2(src, dst)
    return {"counts": dict(per_class), "dynamic": sorted(skipped_dynamic),
            "ignored": sorted(skipped_ignored), "found": len(items)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_DIR, help="where the downloaded dataset is")
    ap.add_argument("--out", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--max-per-class", type=int, default=300,
                    help="images kept per sign (keeps training fast; default 300)")
    ap.add_argument("--dry-run", action="store_true", help="only count, copy nothing")
    args = ap.parse_args()

    if not args.raw.exists() or not any(args.raw.rglob("*")):
        print(f"\nNo dataset found in {args.raw}\n"
              "Run  python setup_dataset.py  to see where to download a free dataset.")
        return 1
    result = prepare(args.raw, args.out, args.max_per_class, args.dry_run)
    if not result["counts"]:
        print(f"Found {result['found']} image(s) but none with a usable class name.\n"
              "Expected folders named A, B, C ... 0-9 ... or a Roboflow YOLO export (data.yaml).")
        return 1
    print(("DRY RUN - nothing copied.\n" if args.dry_run else "Dataset prepared.\n"))
    for category in ("alphabet", "number", "word"):
        rows = sorted((l, n) for (c, l), n in result["counts"].items() if c == category)
        if rows:
            print(f"{category.upper():9} {len(rows)} classes, {sum(n for _, n in rows)} images")
            weak = [f"{l}({n})" for l, n in rows if n < 20]
            if weak:
                print(f"          WARNING: very few images for {', '.join(weak)}")
    if result["dynamic"]:
        print(f"\nSkipped motion letters {result['dynamic']} (need movement, not a still picture).")
    if result["ignored"]:
        print(f"Ignored helper classes {result['ignored']}.")
    print("\nNext:  python training/extract_features.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
