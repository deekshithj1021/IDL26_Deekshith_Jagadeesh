# AUDIT LOG — Operation Cyber-Histology
**Authors:** Deekshith Jagadeesh, Prahas Hegde

**Course:** MAI/IDL SS26 — Final Assignment

**Date:** 09 July 2026

---

## How I Approached This

When I first received the assignment I read through the PDF
carefully to understand what was being asked. The story about
Dr. Vance and the sabotaged code was dramatic but underneath
it I understood the real task — find and fix bugs in four
Python files and rebuild the missing infrastructure.

My first step was to set up the environment. I cloned the
template repository from GitHub and downloaded the data files
from the cloud link. I organized everything into a clean
folder structure with Code and Data folders and opened the
project in VS Code.

Before touching any code I read through each file once to
understand what it was supposed to do. I saw that data.py
loads and splits the data, models.py defines the three neural
network architectures, fit.py handles the training loop, and
train.py ties everything together. I also noticed immediately
that config.json was completely missing even though train.py
was trying to open it.

My approach was simple — try to run the code and fix whatever
crashes first. Then run again and fix the next problem. I kept
doing this until the code ran cleanly from start to finish.

The first attempt to run train.py crashed immediately with
FileNotFoundError because config.json did not exist. I created
a basic config.json with the minimum settings needed to run.

The second attempt crashed with a shape error in the labels.
I had never seen this error before so I investigated by printing
the label tensor shapes. I found they were stored as [N,1]
instead of the [N] that CrossEntropyLoss expects.

The third attempt actually started training but the loss was
not decreasing at all — it kept growing larger every batch.
I went back into fit.py and realised zero_grad was missing.
PyTorch accumulates gradients by default so without resetting
them they explode across batches.

After fixing the obvious crashes I started looking more
carefully at the logic. I noticed the validation data was
included inside the training data — a silent bug that would
never cause a crash but completely invalidates all results.

I then went through models.py very carefully line by line.
I found the activation was set to Identity which does nothing,
ResNet18 was not returning its output, VGGBlock had wrong
padding on 1x1 convolutions, channel counts were never
updating inside the loop, and AlexNet had hardcoded values
that made it incompatible with most datasets.

Some of these bugs I found by reading the code carefully.
Others I only found when the code crashed while trying to
run a specific model on a specific dataset. For example the
AlexNet classifier size bug only appeared when I actually
ran AlexNet and saw the shape mismatch error.

Going through everything I found a total of 15 bugs across
4 files. I also made one important improvement to fit.py
after observing during my first benchmark run that test
accuracy was sometimes much lower than the best validation
accuracy seen during training — because the code was always
using the final epoch weights even when earlier epochs were
much better.

The whole process felt like detective work. Each fix revealed
the next problem and slowly the picture became clearer.
By the end the code ran cleanly on all three models across
all four datasets and produced meaningful results.

---

## Bug Table

| # | File | Category | How It Manifests | Commit |
|---|------|----------|-----------------|--------|
| 1 | fit.py | Gradient failure | Loss explodes, model never learns | 113ac8e |
| 2 | fit.py | Anti-pattern | sum shadows Python built-in sum() | 113ac8e |
| 3 | data.py | Silent logical flaw | Validation data included in training set | b3c5491 |
| 4 | data.py | Runtime crash | Labels shape [N,1] crashes CrossEntropyLoss | b3c5491 |
| 5 | data.py | Runtime crash | Test labels same shape issue as train labels | b3c5491 |
| 6 | models.py | Numerical failure | Identity activation — network cannot learn | 394111c |
| 7 | models.py | Silent bug | ResNet18 forward() returns None silently | 394111c |
| 8 | models.py | Runtime crash | Wrong padding on 1x1 conv in VGGBlock | 394111c |
| 9 | models.py | Runtime crash | VGGBlock never updates channel count | 394111c |
| 10 | models.py | Runtime crash | AlexNet ignores in_channels and num_classes | 394111c |
| 11 | models.py | Runtime crash | AlexNet classifier wrong input size | 394111c |
| 12 | train.py | Infrastructure | config.json missing — script crashes on startup | cfd66ef |
| 13 | train.py | Infrastructure | MPS device ignored — always falls back to CPU | cfd66ef |
| 14 | train.py | Silent logical flaw | Test loader discarded — no final evaluation | cfd66ef |
| 15 | train.py | Numerical failure | drop_rate hardcoded 0.99 — disables 99% neurons | cfd66ef |

---

## Detailed Bug Reports

---

### BUG 01 — Missing zero_grad() in Training Loop
**File:** `fit.py` | **Function:** `train_one_epoch()` | **Commit:** `113ac8e`

**How I found it:**
After creating config.json I tried running train.py for
the first time. The script started but the training loss
was not decreasing at all — it kept growing larger every
batch instead of getting smaller. The model was not
learning anything. I printed the loss values and saw
they were exploding into very large numbers.

I went into fit.py and read through train_one_epoch()
step by step. I saw loss.backward() and optimizer.step()
were called but there was no optimizer.zero_grad() before
them. I remembered from lectures that PyTorch accumulates
gradients by default — without resetting them they keep
adding up across batches and eventually explode.

**What was wrong:**
PyTorch adds new gradients on top of old ones each batch.
Without resetting, gradients from batch 1 carry into batch 2,
batch 2 into batch 3 and so on. They keep growing until they
explode — loss becomes very large and the model learns nothing.

I specifically chose to put zero_grad() BEFORE
loss.backward() and not after optimizer.step(). The
reason is the correct order must always be:
reset → calculate → update. If I put zero_grad() after
step() it would erase the gradients we just calculated
before using them which defeats the purpose.

**Fix:**
Added `self.optimizer.zero_grad()` before `loss.backward()`.

```python
# before
loss.backward()
self.optimizer.step()

# after
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

---

### BUG 02 — Variable sum Shadows Python Built-in
**File:** `fit.py` | **Function:** `train_one_epoch()` | **Commit:** `113ac8e`

**How I found it:**
While I was fixing Bug 01 and reading through
train_one_epoch() carefully I noticed the counter
variable was named sum. I immediately recognised
this was a problem because sum is a built-in Python
function used to add up lists. Using it as a variable
name silently overwrites the built-in for that entire
function scope.

It was not causing a crash here because the built-in
sum() was not called inside this function. But it is
a dangerous anti-pattern — if any future code in
this function tried to use sum() it would fail
unexpectedly with a confusing error.

**What was wrong:**
Python looks up names in local scope first. When you
write `sum = 0` Python creates a local integer variable
that completely hides the built-in `sum()` function.
Any call to sum() in that scope would return 0 instead
of summing a list — silently producing wrong results
without any error message. This is a scoping anti-pattern
that violates Python best practices.

**Fix:**
Renamed variable from `sum` to `total`. Chose total
specifically because it matches the naming used in the
evaluate() method below — making the class consistent.

```python
# before
correct, sum = 0, 0

# after
correct, total = 0, 0
```

---

### BUG 03 — Validation Data Leaks Into Training Set
**File:** `data.py` | **Function:** `get_loaders()` | **Commit:** `b3c5491`

**How I found it:**
After fixing the fit.py bugs training ran without crashing
but something felt wrong. Validation accuracy was
suspiciously high from the very first epoch. I inspected
data.py carefully and printed the sizes of train and val
datasets. I noticed train had ALL samples and val had the
last 10% of those same samples — the two sets were
completely overlapping.

**What was wrong:**
The split index `val_start` was calculated correctly as
90% of total samples. But it was only applied to create
`val_data` — never applied to cut `train_data`. So the
model was training on validation samples and then being
tested on those same samples. Results were artificially
inflated — this is called data leakage and is one of the
most dangerous silent bugs in machine learning.

**Fix:**
Applied `[:val_start]` slice to training data so it only
contains the first 90% with no overlap with validation.

```python
# before
train_data = data_dict['train_images']
train_labels = data_dict['train_labels']

# after
train_data = data_dict['train_images'][:val_start]
train_labels = data_dict['train_labels'][:val_start].squeeze(1)
```

---

### BUG 04 — Wrong Training and Validation Label Shape
**File:** `data.py` | **Function:** `get_loaders()` | **Commit:** `b3c5491`

**How I found it:**
When I first ran train.py after creating config.json it
crashed immediately on the first training batch with
`RuntimeError: 0D or 1D target tensor expected`.
I investigated by printing the label tensor shapes and
found labels stored as shape `[N, 1]` — each label
wrapped in an extra bracket like `[[7], [3], [6]]`
instead of the flat `[7, 3, 6]` that CrossEntropyLoss
expects.

**What was wrong:**
`nn.CrossEntropyLoss` needs labels as a flat 1D tensor
of shape `[N]` to index into the output tensor correctly.
A shape `[N, 1]` is 2D which CrossEntropyLoss interprets
as multi-target classification — a completely different
problem that causes an immediate crash.

**Fix:**
Applied `.squeeze(1)` to training and validation labels.
`squeeze(1)` removes dimension 1 if it has size 1,
converting `[N, 1]` to `[N]`.

```python
# before
train_labels = data_dict['train_labels'][:val_start]
val_labels   = data_dict['train_labels'][val_start:]

# after
train_labels = data_dict['train_labels'][:val_start].squeeze(1)
val_labels   = data_dict['train_labels'][val_start:].squeeze(1)
```

---

### BUG 05 — Wrong Test Label Shape
**File:** `data.py` | **Function:** `get_loaders()` | **Commit:** `b3c5491`

**How I found it:**
After fixing the training and validation label shapes
I realized the test labels had the same problem. The
test dataset was created directly from `data_dict['test_labels']`
without any squeeze — same `[N, 1]` shape issue.
If left unfixed the final test evaluation would also
crash with the same error.

**What was wrong:**
Same root cause as Bug 04 — test labels stored as
`[N, 1]` in the `.pt` files. CrossEntropyLoss needs
`[N]` for evaluation just as much as for training.
The fix needed to be applied consistently to all
three label tensors — train, val and test.

**Fix:**
Applied `.squeeze(1)` to test labels when creating
the test dataset.

```python
# before
test_dataset = TensorDataset(data_dict['test_images'],
               data_dict['test_labels'])

# after
test_dataset = TensorDataset(data_dict['test_images'],
               data_dict['test_labels'].squeeze(1))
```

---

### BUG 06 — Identity Activation Kills Network Learning
**File:** `models.py` | **Global:** `activation_str` | **Commit:** `394111c`

**How I found it:**
After fixing all the data and training bugs the code finally
ran without crashing. But ResNet18 accuracy was terrible —
staying near random chance no matter how many epochs ran.
Train loss was barely decreasing. I went into models.py
and at the very top I saw `activation_str = "Identity"`.
I immediately knew this was wrong — Identity means pass
the input through completely unchanged, which is the same
as having no activation function at all.

**What was wrong:**
Without non-linear activations a deep neural network
collapses mathematically into a single linear transformation
no matter how many layers it has. Every layer just multiplies
by a matrix and stacks of matrix multiplications are still
just one big matrix multiplication. The network physically
cannot learn non-linear patterns like cell boundaries or
lung textures. It was like building an 18-layer network
that behaves like a single straight line.

**Fix:**
Changed activation to ReLU which makes negative values
zero and keeps positive values unchanged — introducing
the non-linearity needed for deep learning.

```python
# before
activation_str = "Identity"

# after
activation_str = "ReLU"
```

---

### BUG 07 — ResNet18 forward() Returns None
**File:** `models.py` | **Class:** `ResNet18` | **Commit:** `394111c`

**How I found it:**
While reading through each model carefully I went through
ResNet18's forward() method line by line. Every stage
processed the data correctly — conv1, stage1 through
stage4, avgpool, flatten. But the very last line was just
`self.classifier(out)` without return. In Python any
function without a return statement gives back None.
The model was computing the final prediction and then
silently throwing it away.

**What was wrong:**
The forward() method is what PyTorch calls to get
predictions from the model. Returning None meant
the loss function received None instead of class scores.
This produced completely invalid gradients silently —
no crash, no warning, just wrong results. It is the
most dangerous type of bug because everything appears
to run normally.

**Fix:**
Added `return` keyword to the last line of forward().

```python
# before
self.classifier(out)

# after
return self.classifier(out)
```

---

### BUG 08 — Wrong Padding on 1x1 Conv in VGGBlock
**File:** `models.py` | **Class:** `VGGBlock` | **Commit:** `394111c`

**How I found it:**
When I tried running VGG16 it crashed with a shape
mismatch error deep inside the network. I traced the
error back to VGGBlock and read through the conv layer
construction carefully. I noticed that `is_config_c_tail`
was correctly detecting when kernel_size should be 1
but then `padding` was always passed as 1 regardless.
I knew that a 1x1 kernel with padding=1 adds a border
of zeros around the image which artificially increases
its spatial dimensions.

**What was wrong:**
A 3x3 convolution needs padding=1 to maintain spatial
dimensions — the kernel overlaps by 1 pixel on each side.
But a 1x1 convolution looks at exactly one pixel at a time
and needs no padding at all. Adding padding=1 to a 1x1
conv makes the output larger than the input which cascades
through subsequent layers causing shape mismatches.

**Fix:**
Made padding conditional on kernel size — zero for 1x1
convolutions and the original padding value for 3x3.

```python
# before
layers.append(nn.Conv2d(current_in_channels, out_channels,
              kernel_size=kernel_size, padding=padding))

# after
pad = 0 if is_config_c_tail else padding
layers.append(nn.Conv2d(current_in_channels, out_channels,
              kernel_size=kernel_size, padding=pad))
```

---

### BUG 09 — VGGBlock Never Updates Channel Count
**File:** `models.py` | **Class:** `VGGBlock` | **Commit:** `394111c`

**How I found it:**
After fixing the padding bug VGG16 still crashed with
a different channel mismatch error. I added print
statements inside VGGBlock to see what was happening
at each conv layer. I saw that `current_in_channels`
was always the same original value even after the first
conv layer had already changed the channel count to
`out_channels`. The loop was never updating the tracker.

**What was wrong:**
Inside a VGGBlock multiple conv layers are stacked.
After the first conv layer transforms the data from
`in_channels` to `out_channels`, the second and third
conv layers need to accept `out_channels` as their input.
But `current_in_channels` stayed at the original
`in_channels` value so conv2 and conv3 tried to process
the wrong number of input channels — causing a crash.

**Fix:**
Added one line at the end of the loop to update
`current_in_channels` after each conv layer.

```python
# before — current_in_channels never updated
for i in range(num_convs):
    layers.append(nn.Conv2d(current_in_channels, out_channels, ...))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))

# after — channel count updated each iteration
for i in range(num_convs):
    layers.append(nn.Conv2d(current_in_channels, out_channels, ...))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    current_in_channels = out_channels
```

---

### BUG 10 — AlexNet Ignores in_channels and num_classes
**File:** `models.py` | **Class:** `AlexNet` | **Commit:** `394111c`

**How I found it:**
When I tried running AlexNet on the chest dataset it
crashed immediately. I looked at AlexNet's __init__
and saw it only accepted `**kwargs` — it never extracted
`in_channels` or `num_classes` from those kwargs. The
first conv layer had `3` hardcoded and the classifier
had `11` hardcoded. Chest is grayscale with 1 channel
and 2 classes — completely incompatible.

**What was wrong:**
AlexNet was built to work with only one specific dataset
configuration. Every other dataset either had a different
number of channels (chest and orgs are grayscale with
1 channel) or different number of classes. Passing
`in_channels=1` in the config had no effect because
AlexNet ignored it and always built a 3-channel network.

**Fix:**
Added explicit `in_channels` and `num_classes` parameters
to the constructor and used them in the network layers.

```python
# before
def __init__(self, **kwargs):
    nn.Conv2d(3, 48, ...)
    nn.Linear(1024, 11)

# after
def __init__(self, in_channels, num_classes, **kwargs):
    nn.Conv2d(in_channels, 48, ...)
    nn.Linear(1024, num_classes)
```

---

### BUG 11 — AlexNet Classifier Wrong Input Size
**File:** `models.py` | **Class:** `AlexNet` | **Commit:** `394111c`

**How I found it:**
After fixing AlexNet to accept the correct parameters
it still crashed with `RuntimeError: mat1 and mat2
shapes cannot be multiplied`. I calculated the actual
feature size manually by tracing through each layer:

```
Input: 64x64
After Conv1 (stride=2): 32x32
After MaxPool1 (stride=2): 16x16
After Conv2: 16x16
After MaxPool2 (stride=2): 8x8
After Conv3, Conv4, Conv5: 8x8
After MaxPool3 (stride=2): 4x4
Channels at end: 192

Flattened: 192 × 4 × 4 = 3072
```

But the classifier had `nn.Linear(2048, 1024)` — wrong.

**What was wrong:**
The hardcoded value 2048 does not match the actual
flattened feature size of 3072 produced by AlexNet's
conv layers for 64x64 input images. This caused an
immediate shape mismatch crash when trying to pass
the features through the first linear layer.

**Fix:**
Changed 2048 to the correct calculated value 3072.

```python
# before
nn.Linear(2048, 1024)

# after
nn.Linear(3072, 1024)
```

---

### BUG 12 — config.json Missing Entirely
**File:** `train.py` | **Function:** `main()` | **Commit:** `cfd66ef`

**How I found it:**
The very first thing that happened when I ran train.py
was a FileNotFoundError — config.json did not exist at all.
The script crashed before doing anything. I looked at the
code and saw `open("config.json", "r")` at the top but
the file was completely deleted. This meant there was no
way to control any settings without hardcoding them directly
into the Python files — exactly the kind of infrastructure
violation the assignment warned about.

**What was wrong:**
Without config.json the entire pipeline was impossible to
run. Even if the file existed, all the important values
like which model to use, which dataset, batch size and
learning rate were either missing or hardcoded. Running
different models or datasets required manually editing
Python code each time — completely against the spirit of
a config-driven pipeline.

**Fix:**
Created config.json from scratch with two sections.
SHARED for settings common to all runs and CONFIGS as
a list of all model/dataset combinations. Updated train.py
to read both sections and loop through all combinations.

```json
{
    "SHARED": {"DATA_PATH": "../Data", "BATCH_SIZE": 64,
               "LEARNING_RATE": 0.001, "DROP_RATE": 0.5},
    "CONFIGS": [
        {"MODEL": "ResNet18", "DATA": "cells",
         "CHANNELS": 3, "NUM_CLASSES": 8, "EPOCHS": 10}
    ]
}
```

---

### BUG 13 — MPS Device Not Detected
**File:** `train.py` | **Function:** `main()` | **Commit:** `cfd66ef`

**How I found it:**
After getting the code running I noticed training was
extremely slow. Each epoch was taking several minutes.
I checked what device was being used and saw it was
running on CPU. I was on a Mac with Apple Silicon which
has a GPU backend called MPS. The original device
detection only checked for CUDA which is an NVIDIA
technology — completely irrelevant on Mac. So it always
fell back to slow CPU even though a GPU was available.

**What was wrong:**
The original one-liner only knew about two options:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Apple Silicon uses MPS (Metal Performance Shaders) not
CUDA. Without checking for MPS the Mac GPU was completely
ignored and training ran 3-5x slower than necessary on CPU.

**Fix:**
Added MPS detection as the first check before CUDA so
the Mac GPU is used automatically when available.

```python
# before
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# after
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

---

### BUG 14 — Test Loader Discarded, No Final Evaluation
**File:** `train.py` | **Function:** `main()` | **Commit:** `cfd66ef`

**How I found it:**
I noticed `get_loaders()` returns three things but the
original code only kept two of them — the test loader
was thrown away using `_`. This meant after all the
training there was no way to actually evaluate the model
on unseen test data. The whole point of training is to
get a final test accuracy but the original code made
that completely impossible.

**What was wrong:**
The test set is the only honest measure of how well the
model generalizes to completely new data. By discarding
the test loader and having no evaluation after training
the pipeline was fundamentally incomplete. Training would
finish and print nothing useful — no way to know if the
model actually learned anything.

**Fix:**
Saved the test loader and added evaluation after training.

```python
# before
train_loader, val_loader, _ = get_loaders(...)
# no test evaluation at all

# after
train_loader, val_loader, test_loader = get_loaders(...)
test_loss, test_acc = trainer.evaluate(test_loader)
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
```

---

### BUG 15 — drop_rate Hardcoded at 0.99
**File:** `train.py` | **Function:** `main()` | **Commit:** `cfd66ef`

**How I found it:**
I read through the model creation line in train.py and
saw `drop_rate=0.99` hardcoded directly in the code.
I immediately knew this was wrong — dropout of 0.99
means 99% of neurons are randomly disabled during each
training step. Only 1% of the network is active at any
time which is far too aggressive for any meaningful
learning. I also noticed `activation_str=None` was being
passed as a kwarg which models do not accept.

**What was wrong:**
With 99% dropout the model has almost no capacity to
process information during training. Each forward pass
uses only 1% of neurons — completely insufficient to
learn complex medical image patterns. Also hardcoding
this value meant it could not be controlled from config
which violated the config-driven design principle.

**Fix:**
Changed to read `DROP_RATE` from the shared config
where the correct value 0.5 is set. Also removed the
invalid `activation_str=None` kwarg since activation
is already controlled by the global variable in models.py.

```python
# before
model = model_class(..., drop_rate=0.99, activation_str=None)

# after
model = model_class(..., drop_rate=shared["DROP_RATE"])
```

---

## Code Improvement Beyond Bug Fixes

### IMPROVEMENT — Best Weight Saving in fit.py
**File:** `fit.py` | **Commit:** `02e2ddf`

**How I discovered this:**
During my first full benchmark run something unexpected
happened with ResNet18 on cells. I watched the validation
accuracy carefully during training and saw it reach 96.34%
at epoch 4. But by epoch 10 it had dropped badly to 92.10%.
When the final test accuracy printed it was only 68.14% —
much worse than what I had seen during training.

I realised the problem immediately. The code was always
using the weights from the very last epoch for test
evaluation. But the last epoch was not the best epoch —
the model had peaked at epoch 4 and then overfit in later
epochs. We were throwing away the best version of the model
and evaluating with a degraded one instead.

This is not a bug in the original code — it ran correctly.
But it was a serious practical problem that made results
unreliable and inconsistent between runs. I decided to
fix it because without it the test accuracy was essentially
random depending on what happened to be the last epoch.

**What I added:**
I added best weight tracking to the fit() method in the
Trainer class. After each epoch if the validation accuracy
is better than anything seen before I save a deep copy
of the model weights. After training finishes I restore
these best weights before returning — so test evaluation
always uses the best model found during training not
the final potentially degraded one.

I used `copy.deepcopy()` specifically rather than a simple
assignment because PyTorch model weights are mutable objects.
A simple assignment would just create another reference to
the same weights — if training continued those weights would
change. deepcopy makes a completely independent snapshot
that stays frozen even as training continues.

```python
# added to fit()
best_val_acc = 0.0
best_weights = None

if val_acc > best_val_acc:
    best_val_acc = val_acc
    # deepcopy — independent snapshot not affected by further training
    best_weights = copy.deepcopy(self.model.state_dict())

# after training loop — restore best weights before evaluation
if best_weights is not None:
    self.model.load_state_dict(best_weights)
    print(f" Restored best weights (val acc: {best_val_acc:.2f}%)")
```

**Measured impact:**
The difference was dramatic and proved the improvement
was necessary:

| Run | Without improvement | With improvement |
|-----|--------------------|--------------------|
| ResNet18/cells | 68.14% | 95.44% |
| VGG16/cells | varies | 94.42% |
| AlexNet/cells | varies | 92.25% |

The +27.3% improvement on ResNet18/cells alone shows
how critical it is to evaluate with the best weights
found during training rather than the arbitrary final
epoch weights.