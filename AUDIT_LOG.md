# AUDIT LOG — Operation Cyber-Histology
**Author:** Deekshith Jagadeesh  
**Date:** 08.06.2026  
**Course:** MAI/IDL SS26 — Final Assignment

---

## Summary
A total of 10 bugs were identified and neutralized across 4 recovered source files.
The bugs span all four failure categories identified by the forensic team:
crashing errors, silent logical flaws, numerical/gradient failures,
and rigid infrastructure issues.

---

## Bug Table

| # | File | Bug Type | How It Manifests | Commit |
|---|------|----------|-----------------|--------|
| 1 | fit.py | Gradient Failure | Loss explodes, model never converges | bc4a616 |
| 2 | data.py | Silent Logical Flaw | Validation data leaks into training set | bc4a616 |
| 3 | data.py | Runtime Crash | Labels have wrong shape, CrossEntropyLoss crashes | bc4a616 |
| 4 | models.py | Numerical Failure | Identity activation kills all non-linearity | bc4a616 |
| 5 | models.py | Silent Bug | ResNet18 forward() returns None | bc4a616 |
| 6 | models.py | Runtime Crash | Wrong padding on 1x1 convolution in VGGBlock | bc4a616 |
| 7 | models.py | Runtime Crash | VGGBlock never updates current_in_channels | 9c45dc9 |
| 8 | models.py | Runtime Crash | AlexNet classifier hardcoded wrong input size | 181f522 |
| 9 | train.py | Runtime Crash | AlexNet ignores in_channels and num_classes | bc4a616 |
| 10 | train.py | Rigid Infrastructure | Hardcoded drop_rate=0.99, missing config.json | bc4a616 |

---

## Detailed Bug Reports

---

### BUG 01 — Missing `zero_grad()` in Training Loop
**File:** `fit.py`  
**Function:** `train_one_epoch()`  
**Commit:** `bc4a616`

**How it manifests:**
Gradients accumulate across every batch instead of being reset.
Loss values explode into very large numbers and the model
fails to learn anything meaningful.

**Root Cause:**
In backpropagation, PyTorch accumulates gradients by default.
`optimizer.zero_grad()` must be called before `loss.backward()`
to reset the gradient buffer at the start of each batch.
Without it, gradients from batch 1 carry over into batch 2,
batch 2 into batch 3, and so on — causing gradient explosion.

**Fix:**
Added `self.optimizer.zero_grad()` before `loss.backward()`.

```python
# Before (broken)
loss.backward()
self.optimizer.step()

# After (fixed)
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

---

### BUG 02 — Validation Data Leaks Into Training Set
**File:** `data.py`  
**Function:** `get_loaders()`  
**Commit:** `bc4a616`

**How it manifests:**
The model silently trains on validation data. Validation
accuracy appears artificially high but the model has not
truly generalised. Results are completely invalid.

**Root Cause:**
`train_data` was assigned the full dataset including the
validation slice. The split index `val_start` was computed
correctly but never applied to the training set — only to
the validation set. Both sets shared the same data.

**Fix:**
Applied `[:val_start]` slice to training data so it only
contains the first 90% of samples.

```python
# Before (broken)
train_data = data_dict['train_images']
train_labels = data_dict['train_labels']

# After (fixed)
train_data = data_dict['train_images'][:val_start]
train_labels = data_dict['train_labels'][:val_start]
```

---

### BUG 03 — Wrong Label Shape Causes CrossEntropyLoss Crash
**File:** `data.py`  
**Function:** `get_loaders()`  
**Commit:** `bc4a616`

**How it manifests:**
RuntimeError: `0D or 1D target tensor expected,
multi-target not supported`. Training crashes immediately
on the first batch.

**Root Cause:**
Labels stored in the `.pt` files have shape `[N, 1]`
(each label wrapped in an extra dimension).
`nn.CrossEntropyLoss` requires flat 1D labels of shape `[N]`.
The extra dimension causes a shape mismatch.

**Fix:**
Applied `.squeeze(1)` to all label tensors to remove
the redundant dimension.

```python
# Before (broken)
train_labels = data_dict['train_labels'][:val_start]
val_labels   = data_dict['train_labels'][val_start:]
test_labels  = data_dict['test_labels']

# After (fixed)
train_labels = data_dict['train_labels'][:val_start].squeeze(1)
val_labels   = data_dict['train_labels'][val_start:].squeeze(1)
test_labels  = data_dict['test_labels'].squeeze(1)
```

---

### BUG 04 — Identity Activation Kills Network Learning
**File:** `models.py`  
**Global variable:** `activation_str`  
**Commit:** `bc4a616`

**How it manifests:**
ResNet18 trains without crashing but learns nothing.
Accuracy stays near random chance regardless of epochs.

**Root Cause:**
`activation_str = "Identity"` means every activation
function in ResNet18 is `nn.Identity()` — a pass-through
that does nothing. Without non-linear activations, the
entire deep network collapses mathematically into a single
linear transformation, no matter how many layers it has.
It cannot learn non-linear patterns.

**Fix:**
Changed activation to ReLU which introduces non-linearity.

```python
# Before (broken)
activation_str = "Identity"

# After (fixed)
activation_str = "ReLU"
```

---

### BUG 05 — ResNet18 forward() Returns None
**File:** `models.py`  
**Class:** `ResNet18`  
**Function:** `forward()`  
**Commit:** `bc4a616`

**How it manifests:**
Silent bug — no crash but the model outputs None instead
of predictions. Loss calculation receives None and produces
completely invalid gradients or silent NaN values.

**Root Cause:**
The last line of `forward()` calls `self.classifier(out)`
but does not return the result. In Python, a function
without a return statement returns None by default.

**Fix:**
Added `return` keyword to the final line.

```python
# Before (broken)
self.classifier(out)

# After (fixed)
return self.classifier(out)
```

---

### BUG 06 — Wrong Padding on 1x1 Convolution in VGGBlock
**File:** `models.py`  
**Class:** `VGGBlock`  
**Function:** `__init__()`  
**Commit:** `bc4a616`

**How it manifests:**
RuntimeError: tensor shape mismatch in deeper VGG layers.
The spatial dimensions grow unexpectedly due to incorrect
padding on 1x1 convolutions.

**Root Cause:**
VGG config C uses a 1x1 convolution for the third conv
in 3-conv blocks. The code correctly detects this but
still applies `padding=1` to the 1x1 kernel. A 1x1
convolution needs no padding — adding padding artificially
increases the spatial dimensions of the feature maps,
causing shape mismatches in subsequent layers.

**Fix:**
Made padding conditional on kernel size.

```python
# Before (broken)
layers.append(nn.Conv2d(current_in_channels, out_channels,
              kernel_size=kernel_size, padding=padding))

# After (fixed)
pad = 0 if is_config_c_tail else padding
layers.append(nn.Conv2d(current_in_channels, out_channels,
              kernel_size=kernel_size, padding=pad))
```

---

### BUG 07 — VGGBlock Does Not Update Channel Count Between Convolutions
**File:** `models.py`  
**Class:** `VGGBlock`  
**Function:** `__init__()`  
**Commit:** `9c45dc9`

**How it manifests:**
RuntimeError on first VGG16 forward pass — channel mismatch
between consecutive conv layers inside the same VGGBlock.

**Root Cause:**
`current_in_channels` is initialized to `in_channels` before
the loop but never updated after each conv layer. So conv2
and conv3 inside the block still expect the original input
channels instead of `out_channels` from the previous conv.
This breaks every multi-conv VGG block.

**Fix:**
Added one line at the end of the loop body to update
the channel count after each conv layer.

```python
# Before (broken)
layers.append(nn.Conv2d(current_in_channels, out_channels, ...))
layers.append(nn.BatchNorm2d(out_channels))
layers.append(nn.ReLU(inplace=True))
# current_in_channels never updated

# After (fixed)
layers.append(nn.Conv2d(current_in_channels, out_channels, ...))
layers.append(nn.BatchNorm2d(out_channels))
layers.append(nn.ReLU(inplace=True))
current_in_channels = out_channels
```

---

### BUG 08 — AlexNet Classifier Input Size Hardcoded Wrong
**File:** `models.py`  
**Class:** `AlexNet`  
**Function:** `__init__()`  
**Commit:** `181f522`

**How it manifests:**
RuntimeError: linear input and weight shapes cannot be
multiplied. AlexNet crashes on first forward pass for
all datasets.

**Root Cause:**
The first Linear layer in AlexNet's classifier hardcodes
`2048` as the input size. But for 64x64 input images,
the actual flattened feature size after the conv layers
is `3072` (192 channels x 4 x 4 spatial).
The wrong number causes an immediate shape mismatch crash.

**Fix:**
Changed the hardcoded `2048` to the correct value `3072`.

```python
# Before (broken)
nn.Linear(2048, 1024)

# After (fixed)
nn.Linear(3072, 1024)
```

---

### BUG 09 — AlexNet Ignores in_channels and num_classes
**File:** `models.py` and `train.py`  
**Class:** `AlexNet`  
**Commit:** `bc4a616`

**How it manifests:**
AlexNet always uses 3 input channels and 11 output classes
regardless of the dataset. Crashes on grayscale datasets
(chest, organs) and produces wrong output dimensions
for all other datasets.

**Root Cause:**
AlexNet's `__init__` only accepted `**kwargs` and never
used `in_channels` or `num_classes` from it. The first
conv layer hardcoded `3` channels and the classifier
hardcoded `11` classes.

**Fix:**
Added explicit `in_channels` and `num_classes` parameters
to AlexNet's constructor and used them in the network.

```python
# Before (broken)
def __init__(self, **kwargs):
    nn.Conv2d(3, 48, ...)
    nn.Linear(1024, 11)

# After (fixed)
def __init__(self, in_channels, num_classes, **kwargs):
    nn.Conv2d(in_channels, 48, ...)
    nn.Linear(1024, num_classes)
```

---

### BUG 10 — Hardcoded drop_rate=0.99 and Missing config.json
**File:** `train.py`  
**Commit:** `bc4a616`

**How it manifests:**
`drop_rate=0.99` drops 99% of neurons making the network
unable to learn. `activation_str=None` passed as kwarg
causes unexpected behaviour. No `config.json` existed
so the script crashed immediately on startup.

**Root Cause:**
Critical hyperparameters were hardcoded instead of being
read from an external configuration file. The entire
infrastructure for config-driven training was missing.

**Fix:**
Created `config.json` with all required hyperparameters.
Updated `train.py` to read `drop_rate` from config.
Removed invalid `activation_str=None` kwarg.

```python
# Before (broken)
model = model_class(..., drop_rate=0.99, activation_str=None)

# After (fixed)
model = model_class(..., drop_rate=config["DROP_RATE"])
```

---

## Required Accuracy Targets

| Dataset | Required | Achieved | Best Model |
|---------|----------|----------|------------|
| cells   | 90%      | 95.50%   | AlexNet    |
| chest   | 87%      | 88.14%   | AlexNet    |
| lesions | 67%      | 73.42%   | VGG16      |
| orgs    | 83%      | 91.24%   | VGG16      |

---

## Code Improvements (Beyond Bug Fixes)

### IMPROVEMENT 01 — Early Stopping and Best Weight Restoration
**File:** `fit.py`  
**Commit:** `b22a8f9`

**Justification:**
Without early stopping, training continued past the optimal
validation accuracy and used degraded weights at the final
epoch. In testing, ResNet18 on cells achieved 97.59% val
accuracy at epoch 17 but ended at 75.20% at epoch 20 — a
22% drop. Early stopping automatically detects when
validation accuracy stops improving (patience=5 epochs)
and restores the best weights found during training.
This is standard practice in production ML pipelines.

**Measured impact on test accuracy:**

| Model | Dataset | Without Early Stopping | With Early Stopping |
|-------|---------|----------------------|---------------------|
| ResNet18 | cells | 74.51% | 95.26% |
| AlexNet | lesions | 62.49% | 70.37% |
| VGG16 | cells | 95.26% | 96.05% |
| AlexNet | cells | 91.03% | 95.50% |

---

### IMPROVEMENT 02 — MPS Device Support for Apple Silicon
**File:** `train.py` and `run_all.py`  
**Commit:** `bc4a616`

**Justification:**
Original code only checked for CUDA GPU. Apple Silicon
Macs use a different backend called MPS. Without this fix
training defaults to CPU which is 3-5x slower on Mac.
Added automatic MPS detection so the pipeline runs
optimally on all hardware without manual changes.

```python
# Before (broken)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# After (fixed)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```