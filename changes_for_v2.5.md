# Changes for v2.5 Migration

This document summarizes the changes made to the `TabPFN-Wide` training code relative to commit `a7f4633a5f4ff53a1deedda82d7b36f70691323f`. These changes should be ported to the v2.5 branch.

## 1. Data Path Fix (Shamir Data)
**File:** `analysis/load_mm_data.py`

**Issue:** The code was incorrectly looking for clinical data (labels for GBM, Breast, Sarcoma) in `benchmark_data/shamir_data/{dataset}/{dataset}`.
**Fix:** Updated to point to `benchmark_data/shamir_data/clinical/clinical/{dataset}`.

```python
# In load_multiomics_benchmark_shamir function:

# OLD
# clinical_data = pd.read_table(os.path.join(dir_path, dataset))

# NEW
clinical_path = os.path.join(shamir_path, "clinical", "clinical", dataset)
clinical_data = pd.read_table(clinical_path)
```

## 2. Infrastructure & Configuration
**Files:** `training/train.sh`, `training/training.sbatch`

**Issue:** Batch size mismatch and single-process execution preventing multi-GPU usage.
**Fix:**
- **Sbatch:** Request 2 GPUs (`--gres=gpu:2`).
- **Train Script:** Launch 2 processes with `torchrun` explicitly.
- **Batch Size:** Set to 8 (effectively 4 per GPU).
- **Prior Batch Size Per GP:** Increase to 32 (`--prior_batch_size_per_gp 32`) to prevent batch skipping due to generator misalignment.

**Changes:**
1.  **`training.sbatch`**: `#SBATCH --gres=gpu:2`
2.  **`train.sh`**:
    ```bash
    torchrun --nproc_per_node=2 --master_port ${MASTER_PORT} ... --batch_size 8 \
        --prior_batch_size_per_gp 32 ...
    ```

## 3. Training Loop Logic (`training/train.py`)

### A. Model Forward Pass Arguments
**Issue:** `PerFeatureTransformer` (v2.0) expects `(x, y)` where `x` is the concatenated sequence. Previous code passed `train_x`, `test_x` keywords.
**Fix:**
```python
# OLD
# pred_logits = self.model(train_x=X_train, train_y=y_train, test_x=X_test)

# NEW
full_x = torch.cat([X_train, X_test], dim=0)
pred_logits = self.model(full_x, y_train)
```

### B. Validation `fit` Argument
**Issue:** `TabPFNClassifier.fit` does not accept a `model` keyword argument.
**Fix:** Inject the model manually into the classifier instance.
```python
# OLD
# clf.fit(X_train_np, y_train_np, model=self.base_model)

# NEW
clf.model = self.base_model
clf.fit(X_train_np, y_train_np)
```

### C. Logging & Monitoring
**Changes:**
-   **Exact Step Logging:** Added `step=self.curr_step` to all `wandb.log` calls to ensure alignment.
-   **Terminal Output:** Added `print` statements with `flush=True` for loss per 10 steps.
-   **WandB Object:** Switched from global `wandb.log` to `self.wandb_obj.log`.
-   **Fix Batch Skip Logging:** Corrected logic to print skipped batch info.

## 4. Missing Utilities (`training/utils.py`)
**Issue:** `NameError: get_linear_added_features not defined`.
**Fix:** Added `get_feature_dependent_noise` and `get_linear_added_features` functions (ported from `analysis/utils.py`).

@torch.no_grad()
def get_feature_dependent_noise(x_tensor, std):
    # The noise std is proportional to the standard deviation of each feature
    stds = x_tensor.std(dim=0, keepdim=True)
    stds[stds == 0] = 1  # Avoid division by zero
    noise = torch.randn_like(x_tensor) * (std * stds)
    return noise


@torch.no_grad()
def get_linear_added_features(x, features_to_be_added, sparsity, noise_std):
    """
    Adds new linear features to the input tensor with controlled sparsity and feature-dependent noise.
    """
    W_sparse = nn.Linear(x.shape[-1], features_to_be_added, bias=False).to(x.device)
    W_sparse.weight.data *= (torch.rand_like(W_sparse.weight) < sparsity).float()
    x = W_sparse(x)

    dependent_noise = get_feature_dependent_noise(x, noise_std)
    x += dependent_noise
    return x.detach()
