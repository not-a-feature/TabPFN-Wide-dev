# Changes for v2.5 Migration

This document summarizes the changes required to port the current working configuration to the `v2.5` branch. The goal is to fix training infrastructure, data loading, and model compatibility issues.

## 1. Data Path Fix (Shamir Data)
**File:** `analysis/load_mm_data.py`

**Issue:** Incorrect clinical data path for GBM, Breast, Sarcoma.
**Fix:** Update `load_multiomics_benchmark_shamir` to use the correct subdirectory.

```python
# In load_multiomics_benchmark_shamir:
# OLD
# clinical_data = pd.read_table(os.path.join(dir_path, dataset))

# NEW
clinical_path = os.path.join(shamir_path, "clinical", "clinical", dataset)
clinical_data = pd.read_table(clinical_path)
```

## 2. Infrastructure Configuration
**Files:** `training/train.sh`, `training/training.sbatch`

**Issue:** Batch size skipping due to single-process execution and small generator groups.
**Fixes:**
1.  **Request 2 GPUs:** In `training.sbatch`, set `#SBATCH --gres=gpu:2`.
2.  **Use 2 Processes:** In `train.sh`, add `--nproc_per_node=2` to `torchrun`.
3.  **Global Batch Size:** In `train.sh`, set `--batch_size 8` (results in 4 per GPU).
4.  **Batch Stability:** In `train.sh`, set `--prior_batch_size_per_gp 1024` to prevent frequent batch skipping caused by generator misalignment.

**Snippet (`train.sh`):**
```bash
torchrun --nproc_per_node=2 --master_port ${MASTER_PORT} "${BASE_DIR_LOCAL}/training/train.py" \
    --batch_size 8 \
    --prior_batch_size_per_gp 1024 \
    ...
```

## 3. Training Loop Logic (`training/train.py`)

### A. Model Forward Pass
**Issue:** `PerFeatureTransformer` expects `(x, y)` positional args.
**Fix:**
```python
# full_x = torch.cat([X_train, X_test], dim=0)
pred_logits = self.model(
    full_x,
    y_train,
)
```

### B. Validation Fit
**Issue:** `TabPFNClassifier.fit` signature mismatch.
**Fix:** Assign `clf.model` directly.
```python
clf.model = self.base_model
clf.fit(X_train_np, y_train_np)
```

### C. Logging
**Fix:**
-   Add `step=self.curr_step` to `self.wandb_obj.log` calls.
-   Print loss to terminal with `flush=True`.

## 4. Missing Utilities (`training/utils.py`)
**Issue:** `get_feature_dependent_noise` and `get_linear_added_features` missing.
**Fix:** Add these functions.

```python
@torch.no_grad()
def get_feature_dependent_noise(x_tensor, std):
    stds = x_tensor.std(dim=0, keepdim=True)
    stds[stds == 0] = 1
    noise = torch.randn_like(x_tensor) * (std * stds)
    return noise

@torch.no_grad()
def get_linear_added_features(x, features_to_be_added, sparsity, noise_std):
    W_sparse = nn.Linear(x.shape[-1], features_to_be_added, bias=False).to(x.device)
    W_sparse.weight.data *= (torch.rand_like(W_sparse.weight) < sparsity).float()
    x = W_sparse(x)
    dependent_noise = get_feature_dependent_noise(x, noise_std)
    x += dependent_noise
    return x.detach()
```
