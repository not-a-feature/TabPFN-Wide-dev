import os
import numpy as np

# Ensure local package is importable when running from repo root
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

from tabpfnwide.classifier import TabPFNWideClassifier

# Path to the requested checkpoint
CHECKPOINT_PATH = os.path.join(
    CURRENT_DIR,
    "checkpoints",
    "5_AddFeat8000_NEst8_Group3",
    "20251218_123325_final_icy-sea-9.pt",
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
        model_path=CHECKPOINT_PATH,  # Use the correct n_estimators and features_per_group for this checkpoint (check config.json)
        n_estimators=8,
        features_per_group=3,
        ignore_pretraining_limits=True,
        device="cpu",
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
