"""Step 4 of 4 (optional): check how good a saved model really is.

    python training/evaluate_model.py --category alphabet

Re-creates the SAME held-out test pictures used during training and prints
accuracy, a per-sign report and the most common mix-ups (e.g. "M is often read as N").
The report is also written to models/<category>_evaluation.txt
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.constants import CATEGORY_ALIASES, MODEL_FILES  # noqa: E402
from core.paths import MODELS_DIR  # noqa: E402
from training.train_model import load_dataset, split_data  # noqa: E402


def evaluate(category: str, models_dir: Path = MODELS_DIR) -> str | None:
    import joblib
    from sklearn.metrics import accuracy_score, classification_report
    model_path = models_dir / MODEL_FILES[category]
    if not model_path.exists():
        return None
    loaded = load_dataset(category)
    if loaded is None:
        return None
    bundle = joblib.load(model_path)
    X, y, _ = loaded
    _, X_test, _, y_test = split_data(X, y, bundle.get("test_size", 0.2))
    pred = bundle["model"].predict(X_test)
    lines = [f"Category: {category}   algorithm: {bundle.get('algo')}   trained: {bundle.get('trained_at')}",
             f"Accuracy on {len(y_test)} unseen samples: {accuracy_score(y_test, pred) * 100:.1f}%", "",
             classification_report(y_test, pred, zero_division=0)]
    confusions = Counter((a, b) for a, b in zip(y_test, pred) if a != b)
    if confusions:
        lines.append("Most common mix-ups (true -> predicted):")
        lines += [f"  {a} -> {b}: {n} times" for (a, b), n in confusions.most_common(8)]
    else:
        lines.append("No mistakes on the test set (small datasets can be misleadingly perfect).")
    report = "\n".join(lines)
    try:
        (models_dir / f"{category}_evaluation.txt").write_text(report, encoding="utf-8")
    except OSError:
        pass
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="alphabet")
    args = ap.parse_args()
    cat = CATEGORY_ALIASES.get(args.category)
    if cat is None:
        print(f"Unknown category '{args.category}'.")
        return 1
    report = evaluate(cat)
    if report is None:
        print(f"No trained model or data for '{cat}'. Run train_model.py first.")
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
