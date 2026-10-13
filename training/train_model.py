"""Step 3 of 4: train the local gesture classifiers.

    python training/train_model.py                       # all categories that have data
    python training/train_model.py --category alphabet --algo svm

Categories:  alphabet | numbers | words | custom | dynamic | all
Algorithms:  rf (Random Forest, default) | svm | knn

Saved to ./models/ :
    asl_alphabet_model.pkl  numbers_model.pkl  words_model.pkl
    custom_model.pkl (your recorded still gestures)  dynamic_model.pkl (recorded motions)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.constants import CATEGORY_ALIASES, FEATURE_VERSION, MODEL_FILES  # noqa: E402
from core.paths import CUSTOM_DIR, MODELS_DIR, PROCESSED_DIR  # noqa: E402

SEED = 42


def load_dataset(category: str, processed: Path = PROCESSED_DIR,
                 custom_dir: Path = CUSTOM_DIR) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Return (X, y, two_hand) or None when there is no data for this category."""
    if category in ("alphabet", "number", "word"):
        path = processed / f"features_{category}.npz"
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=False)
        return data["X"], data["y"], bool(data["two_hand"])
    folder = custom_dir / ("dynamic" if category == "dynamic" else "static")
    if not folder.exists():
        return None
    X, y = [], []
    for label_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        for f in sorted(label_dir.glob("*.npy")):
            X.append(np.load(f))
            y.append(label_dir.name.upper())
    if not X:
        return None
    lengths = {len(v) for v in X}
    if len(lengths) != 1:
        raise ValueError(f"Recorded samples have different sizes {sorted(lengths)} - re-record them.")
    return np.array(X, dtype=np.float32), np.array(y), len(X[0]) > 100 and category != "dynamic"


def make_classifier(algo: str):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    if algo == "svm":
        clf = SVC(kernel="rbf", C=10, probability=True, random_state=SEED)
    elif algo == "knn":
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
    else:
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
    return make_pipeline(StandardScaler(), clf)           # StandardScaler = the "normalisation" step


def split_data(X, y, test_size: float = 0.2):
    from sklearn.model_selection import train_test_split
    counts = {label: int(np.sum(y == label)) for label in set(y)}
    stratify = y if min(counts.values()) >= 2 else None
    return train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=stratify)


def train_category(category: str, algo: str = "rf", test_size: float = 0.2,
                   processed: Path = PROCESSED_DIR, custom_dir: Path = CUSTOM_DIR,
                   models_dir: Path = MODELS_DIR, verbose: bool = True) -> dict | None:
    import joblib
    from sklearn.metrics import accuracy_score, classification_report
    loaded = load_dataset(category, processed, custom_dir)
    if loaded is None:
        if verbose:
            print(f"[{category}] no training data found - skipped.")
        return None
    X, y, two_hand = loaded
    classes = sorted(set(y))
    if len(classes) < 2:
        if verbose:
            print(f"[{category}] need at least 2 different gestures, found {classes} - skipped.")
        return None
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)
    model = make_classifier(algo)
    model.fit(X_train, y_train)
    accuracy = float(accuracy_score(y_test, model.predict(X_test)))
    if verbose:
        print(f"\n=== {category.upper()}  ({len(X)} samples, {len(classes)} classes, {algo}) ===")
        print(f"Test accuracy: {accuracy * 100:.1f}%  (on {len(y_test)} pictures the model never saw)")
        print(classification_report(y_test, model.predict(X_test), zero_division=0))
    bundle = {"model": model, "labels": [str(c) for c in model.classes_], "two_hand": two_hand,
              "n_features": int(X.shape[1]), "category": category, "algo": algo,
              "accuracy": accuracy, "test_size": test_size, "feature_version": FEATURE_VERSION,
              "trained_at": datetime.now().isoformat(timespec="seconds"), "n_samples": int(len(X))}
    models_dir.mkdir(parents=True, exist_ok=True)
    out = models_dir / MODEL_FILES[category]
    joblib.dump(bundle, out)
    if verbose:
        print(f"Saved model -> {out}")
    return bundle


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="all")
    ap.add_argument("--algo", choices=["rf", "svm", "knn"], default="rf")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    if args.category == "all":
        cats = ["alphabet", "number", "word", "custom", "dynamic"]
    else:
        cat = CATEGORY_ALIASES.get(args.category)
        if cat is None:
            print(f"Unknown category '{args.category}'.")
            return 1
        cats = [cat]
    trained = [c for c in cats if train_category(c, args.algo, args.test_size)]
    if not trained:
        print("\nNothing was trained. Prepare data first:\n"
              "  python training/prepare_dataset.py\n  python training/extract_features.py")
        return 1
    print("\nDone. Restart the app - 'Demo Mode' disappears once models are found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
