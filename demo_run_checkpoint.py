import os
import numpy as np

# Ensure local package is importable when running from repo root
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

from tabpfnwide.classifier import TabPFNWideClassifier

# Path to the requested checkpoint
# Path to the requested checkpoint
CHECKPOINT_PATH = "/weka/pfeifer/ppu738/TabPFN-Wide/checkpoints/0_AddFeat1500_NEst1_Group1/20260116_182343_final_genial-firebrand-13.pt"

if __name__ == "__main__":
    if not os.path.isfile(CHECKPOINT_PATH):
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}. Please ensure the path is correct or accessible.")

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
        n_estimators=1,
        features_per_group=1,
        ignore_pretraining_limits=True,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
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
