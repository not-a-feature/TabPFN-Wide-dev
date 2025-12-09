# TabPFN Feature Grouping and NaN Handling

This document summarizes the `features_per_group` parameter and NaN handling mechanisms in TabPFN.

---

## 1. Feature Grouping (`features_per_group`)

### What It Does

The `features_per_group` parameter controls **how individual input features are grouped together** before being processed by the attention mechanism. It's defined in [`config.py`](file:///c:/Users/jules/Documents/PrivateAIM/TabPFN-Wide/TabPFN/src/tabpfn/architectures/base/config.py#L30-L32):

```python
features_per_group: PositiveInt = 2
"""If > 1, the features will be grouped into groups of this size and the attention
is across groups."""
```

### Effect on Feature Count and Attention

When `features_per_group > 1`, the model:

1. **Pads features** to a multiple of `features_per_group` (with zeros)
2. **Reshapes** the input so that every `features_per_group` features become one "token" for attention
3. **Reduces the sequence length** for the attention mechanism

**Example:** With 10 input features and `features_per_group=2`:
- The model operates on **5 groups** (tokens) instead of 10 individual features
- Each group contains 2 features that are processed together by the encoder

### Key Locations in the Codebase

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | 30-32 | Default value (`2`) and documentation |
| `transformer.py` | 185 | Stored on the model: `self.features_per_group = config.features_per_group` |
| `transformer.py` | 358-361 | Padding to ensure features are a multiple of `features_per_group` |
| `transformer.py` | 378-384 | Reshapes input using `einops.rearrange` to group features |
| `transformer.py` | 400-410 | Adjusts categorical indices for grouped features |
| `__init__.py` | 83 | Passes to encoder: `num_features=config.features_per_group` |
| `transformer.py` | 269 | Backward compatibility: defaults to `1` for old checkpoints |

### How It Works in `forward()`

The core logic is in `transformer.py` lines 355-384:

```python
# Step 1: Pad to multiple of features_per_group
missing_to_next = (
    self.features_per_group - (num_features_ % self.features_per_group)
) % self.features_per_group

if missing_to_next > 0:
    x[k] = torch.cat((x[k], torch.zeros(..., missing_to_next, ...)), dim=-1)

# Step 2: Reshape to group features together
x[k] = einops.rearrange(
    x[k],
    "s b (f n) -> b s f n",
    n=self.features_per_group,
)  # s b features -> b s #groups #features_per_group
```

This transforms, for example:
- Input: `(seq_len=100, batch=32, features=10)` with `features_per_group=2`
- Output: `(batch=32, seq_len=100, groups=5, features_per_group=2)`

### How It's Set During Training vs. Inference

**Training:**
- Set via the `ModelConfig` when the model is created
- The default value is **2** (see `config.py` line 30)
- Saved as part of the model checkpoint

**Inference:**
- Loaded from the checkpoint's saved config
- For **old checkpoints** without this parameter, it defaults to **1** via `__setstate__` (line 269):
  ```python
  def __setstate__(self, state: dict[str, Any]) -> None:
      state.setdefault("features_per_group", 1)
  ```

### Impact on Encoder Construction

When building the model, `features_per_group` is passed to `get_encoder()` as `num_features` (`__init__.py` line 83):

```python
encoder=get_encoder(
    num_features=config.features_per_group,  # <-- Here
    ...
)
```

This means the **input encoder (Linear or MLP) is sized to accept `features_per_group` input dimensions per token**, not the total number of features. The encoder processes each group independently.

### Summary

| Aspect | Details |
|--------|---------|
| **Purpose** | Reduce attention complexity by processing features in groups |
| **Default** | `2` |
| **Effect** | `n` features become `ceil(n / features_per_group)` attention tokens |
| **Padding** | Zero-padding if features aren't evenly divisible |
| **Encoder size** | Encoder input dim = `features_per_group` (not total features) |
| **Backward compat** | Old checkpoints default to `1` (no grouping) |

---

## 2. Can You Run Inference with Different Grouping?

**No, you cannot safely run inference with a different `features_per_group` than what the model was trained with.**

### Why Not?

The input encoder (Linear or MLP) is constructed with `num_features=config.features_per_group`:

```python
# From __init__.py line 82-83
encoder=get_encoder(
    num_features=config.features_per_group,
    ...
)
```

This means the encoder's **weight matrix dimensions** are fixed at training time:

| `features_per_group` | Encoder Input Dim | 
|----------------------|-------------------|
| 1 | 1 (or 2 with NaN handling) |
| 2 | 2 (or 4 with NaN handling) |
| 3 | 3 (or 6 with NaN handling) |

If you trained with `features_per_group=2`, the encoder expects 2 features per group. Changing to `features_per_group=3` at inference would cause a **shape mismatch**.

**Bottom line:** The grouping is "baked into" the model architecture at training time. Inference must use the same `features_per_group` value.

---

## 3. NaN Handling

TabPFN handles NaN (and infinite) values through the **`NanHandlingEncoderStep`** class located at `encoders.py` lines 612-680.

### How It Works

The `NanHandlingEncoderStep` performs two main operations:

#### 3.1 Creates NaN Indicator Features (if `keep_nans=True`)

It produces a separate tensor (`nan_indicators`) that encodes the location and type of missing/infinite values:

| Value Type | Indicator Value |
|------------|-----------------|
| NaN | `-2.0` |
| +Inf | `+2.0` |
| -Inf | `+4.0` |
| Normal value | `0.0` |

```python
nans_indicator = (
    torch.isnan(x) * -2.0  # nan_indicator
    + torch.logical_and(torch.isinf(x), torch.sign(x) == 1) * 2.0  # inf_indicator
    + torch.logical_and(torch.isinf(x), torch.sign(x) == -1) * 4.0  # neg_inf_indicator
)
```

#### 3.2 Imputes NaN/Inf Values with Feature Means

During the `_fit` phase, it computes the mean of each feature from the **training rows only** (`[:single_eval_pos]`):

```python
def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs):
    self.feature_means_ = torch_nanmean(x[:single_eval_pos], axis=0, include_inf=True)
```

During `_transform`, it replaces NaN and Inf values with these computed means:

```python
nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
x[nan_mask] = self.feature_means_.unsqueeze(0).expand_as(x)[nan_mask]
```

### Where It's Used

The `NanHandlingEncoderStep` is integrated into both the **X encoder** and **Y encoder**:

**For X (Features)** — `__init__.py` lines 140-152:

```python
encoder_steps += [NanHandlingEncoderStep(keep_nans=nan_handling_enabled)]

if nan_handling_enabled:
    inputs_to_merge["nan_indicators"] = {"dim": num_features}
```

When `nan_handling_enabled=True` (default), the encoder produces:
- `main`: The imputed feature values
- `nan_indicators`: The indicator tensor showing where NaNs/Infs were

Both are **concatenated** and fed to the linear/MLP encoder, **doubling the input dimension** (from `features_per_group` to `2 * features_per_group`).

**For Y (Targets)** — `__init__.py` lines 211-213:

```python
if nan_handling_y_encoder:
    steps += [NanHandlingEncoderStep()]
    inputs_to_merge += [{"name": "nan_indicators", "dim": num_inputs}]
```

### Impact on Encoder Input Dimension

Because of NaN handling, the **actual input dimension** to the linear/MLP encoder is:

| NaN Handling | Encoder Input Dim |
|--------------|-------------------|
| **Enabled** (default) | `2 * features_per_group` |
| Disabled | `features_per_group` |

### Configuration Options

From `config.py`:

```python
nan_handling_enabled: Literal[True] = True      # For X features
nan_handling_y_encoder: Literal[True] = True    # For Y targets
```

Both are currently **locked to `True`** (using `Literal[True]`) in the base configuration.

### Known Bug

There's a noted bug in the code at line 667:

```python
# TODO: There is a bug here: The values arriving here are already mapped 
# to nan if they were inf before
```

This means infinity values might already be converted to NaN by earlier preprocessing steps, so the `+Inf` and `-Inf` indicators may not work correctly.

### Data Flow Diagram

```
Input Features (with NaNs/Infs)
         │
         ▼
┌─────────────────────────────────┐
│   NanHandlingEncoderStep        │
│                                 │
│  1. Compute feature means       │
│     (from training rows)        │
│                                 │
│  2. Create nan_indicators:      │
│     NaN → -2.0, +Inf → 2.0      │
│     -Inf → 4.0, normal → 0.0    │
│                                 │
│  3. Replace NaN/Inf with means  │
└─────────────────────────────────┘
         │
         ▼
  [imputed_x, nan_indicators]
         │
         ▼
  Concatenate → Linear/MLP Encoder
```

---

## 4. What Happens When a Column is All-NaN?

The `torch_nanmean` function at `encoders.py` lines 59-90 is **designed for this case**:

```python
def torch_nanmean(...):
    """Computes the mean of a tensor over a given dimension, ignoring NaNs.

    Designed for stability: If all inputs are NaN, the mean will be 0.0.
    """
    nan_mask = torch.isnan(x)
    # ...
    num = torch.where(nan_mask, 0, 1).sum(axis=axis)  # Count of non-NaN values
    value = torch.where(nan_mask, 0, x).sum(axis=axis)  # Sum of non-NaN values
    
    return value / num.clip(min=1.0)  # <-- Key line
```

### The Key: `num.clip(min=1.0)`

| Scenario | `num` | `value` | Result |
|----------|-------|---------|--------|
| Normal column | count of non-NaN | sum of values | proper mean |
| **All-NaN column** | **0** | **0** | **0.0 / 1.0 = 0.0** |

The `.clip(min=1.0)` ensures:
1. **No division by zero** (which would produce NaN or Inf)
2. **All-NaN features get a mean of 0.0**

### Consequence During Transform

When `_transform` runs, all the NaN values in that column get replaced with `0.0`:

```python
x[nan_mask] = self.feature_means_.unsqueeze(0).expand_as(x)[nan_mask]
# All NaNs in that column → 0.0
```

### Summary

**All-NaN columns are imputed with `0.0`** — a sensible fallback that avoids crashes and keeps the data numerically stable. The model also receives the `nan_indicators` tensor showing `-2.0` for every position in that column, so it knows these are all missing values.
