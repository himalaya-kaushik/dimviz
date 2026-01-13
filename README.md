# TorchDimViz 🔎

**Stop guessing tensor shapes. Start seeing them.**

`torchdimviz` is a robust execution tracer for PyTorch. It sits at the lowest level of the PyTorch dispatcher to intercept and log every operation that manipulates your data—whether it's a layer, a reshape, a slice, or an in-place arithmetic operation.

Unlike other visualizers that try to draw complex graphs, `torchdimviz` gives you a simple, linear **execution log** showing exactly how your tensor shapes transform step-by-step.

## Installation

pip install torchdimviz

## Quick Start

```
import torch
import torch.nn as nn
from torchdimviz import DimViz

model = nn.Sequential(
    nn.Conv2d(3, 16, 3),
    nn.Flatten(),
    nn.Linear(16 * 30 * 30, 10)
)

data = torch.randn(1, 3, 32, 32)

print("--- Debugging Forward Pass ---")
with DimViz():
    model(data)
```

## Sample Output

The tool generates a clean ASCII table in your terminal:

```

[DimViz] 🟢 Tracking Started...
+------+-----------------+--------------------+-------------------+
| Step | Operation       | Input Shape        | Output Shape      |
+======+=================+====================+===================+
|    1 | convolution     | (1, 3, 32, 32)     | (1, 16, 30, 30)   |
|    2 | flatten         | (1, 16, 30, 30)    | (1, 14400)        |
|    3 | linear_proj     | (1, 14400)         | (1, 10)           |
+------+-----------------+--------------------+-------------------+
[DimViz] 🔴 Tracking Finished.

```

## Modes

`torchvista` comes with a smart filter to keep your logs readable.

Default Mode (`verbose=True`)

Shows everything. This includes activations (`sigmoid, relu`), element-wise math (`add, mul`), and internal operations inside black-box modules like LSTMs or Transformers.

```
with DimViz(verbose=True):
    model(x)
```

Concise Mode (`verbose=False`)

Only shows operations that change the tensor shape (`like view, permute, slice`) or major layers (`conv, linear`). Useful for quickly debugging shape mismatch errors.

```
with DimViz(verbose=False):
    model(x)
```

## Under the hood

Most PyTorch visualizers use torch.fx (which fails on dynamic loops) or "Hooks" (which miss functional operations like view and slice).

TorchDimViz uses **torch_dispatch**.
This allows it to intercept operations at the C++ kernel level. This means it works seamlessly with:

✔️ CNNs, RNNs, LSTMs, Transformers

✔️ Dynamic Control Flow (for loops, if statements)

✔️ In-place Operations (x.add\_())

✔️ Slicing & Indexing (x[:, -1])

✔️ Arbitrary Reshaping (x.view(b, h, w))

## Requirements

Python 3.8+
PyTorch >= 2.0.0
Tabulate
Rich (optional, for colored output)
