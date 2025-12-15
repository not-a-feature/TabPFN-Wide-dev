import os
import numpy as np

# Ensure local package is importable when running from repo root
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

from tabpfnwide.classifier import TabPFNWideClassifier

# Path to the requested checkpoint
CHECKPOINT_PATH = os.path.join(
    CURRENT_DIR,
    "checkpoints",
    "20251212_003233_final_neat-serenity-5.pt",
)

if __name__ == "__main__":
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    # Small, linearly separable toy dataset
    X = np.array(
        [
            [0.0, 0.0, 0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.0, 0.2, 0.1, 0.4],
            [1.0, 1.0, 1.0, 1.2, 1.1, 1.3],
            [1.1, 0.9, 1.0, 1.3, 1.2, 1.4],
            [0.0, 0.2, 0.1, 0.0, 0.2, 0.1],
            [0.9, 1.1, 1.0, 1.1, 1.2, 1.0],
            [0.2, 0.1, 0.2, 0.1, 0.3, 0.2],
            [1.2, 1.0, 1.1, 1.0, 1.1, 1.2],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)

    clf = TabPFNWideClassifier(
        model_path=CHECKPOINT_PATH,
        device="cpu",  # switch to "cuda" if available and desired
        n_estimators=8,
        features_per_group=1,  # grouping = 3
        ignore_pretraining_limits=True,
    )

    clf.fit(X, y)
    probas = clf.predict_proba(X)
    preds = clf.predict(X)

    print("Predicted probabilities:\n", probas)
    print("Predicted labels:\n", preds)
    if np.array_equal(preds, y):
        print("MATCH: Predictions match the true labels.")
    else:
        print("NOT MATCH: Predictions do not match the true labels.")
