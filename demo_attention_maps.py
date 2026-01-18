import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabpfnwide.classifier import TabPFNWideClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
feature_names = dataset.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X = X_train[:20]
y = y_train[:20]

print("Load model")

clf = TabPFNWideClassifier(
    model_name="wide-v2-5k",
    device="cpu",
    n_estimators=1,  # Only works with 1 estimator and no grouping.
    features_per_group=1,
    save_attention_maps=True,
)


print("Fitting and predicting...")
clf.fit(X, y)
proba = clf.predict_proba(X)

print("Get attention maps")
maps = clf.get_attention_maps()

# Average across layers for a summary view
avg_map = np.mean(maps, axis=0)
print(f"Attention Map Shape: {avg_map.shape}")

plt.figure(figsize=(10, 8))
plt.imshow(avg_map, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Attention Weight")
plt.title("Average Attention Map (Averaged across Layers)")
plt.xlabel("Key Position")
plt.ylabel("Query Position")
plt.show()

importance = avg_map.sum(axis=0)  # total attention received by each key (feature)

# Filter out padding tokens (indices >= number of actual features)
# The model may pad inputs (e.g. to 64 or 128), and these padding tokens can sometimes
# act as "attention sinks" or global context holders. We exclude them to focus on real features.
n_features = len(feature_names)
valid_indices = [i for i in range(len(importance)) if i < n_features]
valid_importance = importance[valid_indices]

# Re-sort based on valid importance
# Note: valid_indices[i] maps back to original index, so we need to be careful.
# Simplest: create pairs of (index, score) and sort
feature_scores = [(i, importance[i]) for i in range(n_features)]
feature_scores.sort(key=lambda x: x[1], reverse=True)

top5 = feature_scores[:5]

print("Top 5 feature indices (most important first):")
for rank, (idx, score) in enumerate(top5, 1):
    name = feature_names[idx]
    print(f"{rank}: {name} (score={float(score):.6f})")

plt.figure(figsize=(8, 4))
# Use feature names if available, otherwise fallback to index
labels = [feature_names[idx] for idx, _ in top5]
scores = [score for _, score in top5]
sns.barplot(x=labels, y=scores)
plt.title("Top 5 feature importances (from avg_map)")
plt.ylabel("Importance (sum of attention received)")
plt.xlabel("Feature")
plt.show()
