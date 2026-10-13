"""Step 2 of 4: turn the prepared pictures into hand-landmark numbers.

    python training/extract_features.py                    # all categories found
    python training/extract_features.py --category alphabet --augment 2

For every image MediaPipe finds the hand and we save the normalised landmark
features (see vision/feature_extractor.py). Pictures where no hand is found are
skipped and counted. The result is a small .npz file per category:
    dataset/processed/features_alphabet.npz   (X = features, y = labels)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.constants import CATEGORY_ALIASES, IMAGE_EXTENSIONS, PROCESSED_DIRS  # noqa: E402
from core.paths import PROCESSED_DIR  # noqa: E402
from vision.feature_extractor import extract_features  # noqa: E402
from vision.hand_detector import HandDetector  # noqa: E402


def features_path(category: str, processed: Path = PROCESSED_DIR) -> Path:
    return processed / f"features_{category}.npz"


def extract_category(category: str, processed: Path, augment: int, two_hand: bool | None,
                     seed: int = 42) -> tuple[int, dict[str, int]]:
    import cv2
    folder = processed / PROCESSED_DIRS[category]
    images = [(p, p.parent.name) for p in sorted(folder.rglob("*"))
              if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        return 0, {}
    detector = HandDetector(2, 0.5, 0.5, static_image_mode=True)
    if not detector.available:
        raise RuntimeError(detector.error)

    detections = []                        # (label, hands)
    failed: dict[str, int] = {}
    for n, (path, label) in enumerate(images, 1):
        img = cv2.imread(str(path))
        hands = detector.detect(img) if img is not None else []
        if hands:
            detections.append((label, hands))
        else:
            failed[label] = failed.get(label, 0) + 1
        if n % 200 == 0:
            print(f"  {category}: {n}/{len(images)} pictures processed…")
    detector.close()
    if not detections:
        return 0, failed

    if two_hand is None:                  # decide automatically: many 2-hand pictures?
        two_hand = sum(len(h) == 2 for _, h in detections) / len(detections) > 0.1
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, hands in detections:
        feats = extract_features(hands, two_hand=two_hand)
        X.append(feats)
        y.append(label)
        for _ in range(augment):          # tiny random jitter = extra training samples
            X.append(feats + rng.normal(0, 0.01, feats.shape).astype(np.float32))
            y.append(label)
    np.savez_compressed(features_path(category, processed), X=np.array(X, dtype=np.float32),
                        y=np.array(y), two_hand=np.array(two_hand))
    return len(X), failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="all", help="alphabet | numbers | words | all")
    ap.add_argument("--augment", type=int, default=1, help="extra jittered copies per picture (default 1)")
    ap.add_argument("--two-hand", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    args = ap.parse_args()

    cats = list(PROCESSED_DIRS) if args.category == "all" else [CATEGORY_ALIASES.get(args.category, "")]
    two = {"auto": None, "yes": True, "no": False}[args.two_hand]
    done = 0
    for cat in cats:
        if cat not in PROCESSED_DIRS:
            print(f"Unknown category: {args.category}")
            return 1
        try:
            count, failed = extract_category(cat, args.processed, args.augment, two)
        except RuntimeError as exc:
            print(f"\nCannot extract features: {exc}")
            return 1
        if count == 0:
            print(f"{cat}: nothing to do (no pictures or no hands found).")
            continue
        done += 1
        lost = sum(failed.values())
        print(f"{cat}: saved {count} samples -> {features_path(cat, args.processed).name}"
              f"  ({lost} pictures had no detectable hand)")
    if not done:
        print("\nNo features created. Did you run prepare_dataset.py first?")
        return 1
    print("\nNext:  python training/train_model.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
