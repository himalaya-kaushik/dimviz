# DimViz 🔍

[![PyPI version](https://badge.fury.io/py/dimviz.svg)](https://badge.fury.io/py/dimviz)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**A lightweight debugging tool for tracking tensor shape transformations in PyTorch models.**

Stop guessing tensor shapes—see them flow through your model in real-time! DimViz helps you debug shape mismatches, understand model architecture, and optimize tensor operations.

## ✨ Features

- 🎯 **Zero Code Changes** - Context manager and decorator patterns
- 📊 **Rich Visualization** - Beautiful terminal tables (with optional Rich library)
- 🔍 **Smart Filtering** - Track only what matters
- 💾 **Multiple Export Formats** - JSON, CSV, TXT
- 📈 **Memory Tracking** - See memory allocation per operation
- 🎨 **Friendly Names** - Human-readable operation names
- ⚡ **Performance Aware** - Minimal overhead with smart filtering

## 🚀 Installation

```bash
pip install dimviz
```

For enhanced visualization with colors and styling:

```bash
pip install dimviz[rich]
```

## 📖 Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
from dimviz import DimViz

# Your model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Track shape transformations
x = torch.randn(32, 784)

with DimViz():
    output = model(x)
```

**Output:**

```
[DimViz] 🟢 Tracking Started...
[DimViz] 🔴 Tracking Finished. (Elapsed: 0.02s)

╭──────┬────────────┬─────────────┬──────────────╮
│ Step │ Operation  │ Input Shape │ Output Shape │
├──────┼────────────┼─────────────┼──────────────┤
│    1 │ linear     │ (32, 784)   │ (32, 256)    │
│    2 │ relu       │ (32, 256)   │ (32, 256)    │
│    3 │ linear     │ (32, 256)   │ (32, 10)     │
╰──────┴────────────┴─────────────┴──────────────╯

[DimViz] 📊 Summary:
  • Total Operations: 3
  • Unique Operations: 2
  • Logged Operations: 3
```

### Decorator Usage

```python
from dimviz import visualize

@visualize()
def train_step(model, batch):
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    return loss

loss = train_step(model, batch)
```

## 🎛️ Advanced Features

### Track Only Shape Changes

When you only care about shape transformations:

```python
with DimViz(verbose=False):
    output = model(x)
```

This filters out operations that don't change tensor shapes (like activation functions on the same tensor).

### Memory Tracking

Monitor memory allocation per operation:

```python
with DimViz(track_memory=True):
    output = model(x)
```

**Output includes memory columns:**

```
╭──────┬───────────┬─────────────┬──────────────┬─────────┬──────────╮
│ Step │ Operation │ Input Shape │ Output Shape │ Mem In  │ Mem Out  │
├──────┼───────────┼─────────────┼──────────────┼─────────┼──────────┤
│    1 │ linear    │ (32, 784)   │ (32, 256)    │ 0.10MB  │ 0.03MB   │
│    2 │ relu      │ (32, 256)   │ (32, 256)    │ 0.03MB  │ 0.03MB   │
│    3 │ linear    │ (32, 256)   │ (32, 10)     │ 0.03MB  │ 0.00MB   │
╰──────┴───────────┴─────────────┴──────────────┴─────────┴──────────╯

[DimViz] 📊 Summary:
  • Peak Memory: 0.10MB
  • Total Memory Delta: -0.07MB
```

### Filter Specific Operations

Focus on particular operation types:

```python
with DimViz(filter_ops=['conv2d', 'matmul', 'linear']):
    output = model(x)
```

### Export Logs

Save your debugging session for later analysis:

```python
from dimviz import export_log

with DimViz() as viz:
    output = model(x)

# Export to various formats
export_log(viz.get_log(), 'debug_log.json')
export_log(viz.get_log(), 'debug_log.csv')
export_log(viz.get_log(), 'debug_log.txt')
```

### Compare Model Runs

Compare shape flows between different model versions:

```python
from dimviz.exporter import compare_logs

# First model
with DimViz() as viz1:
    output1 = model_v1(x)

# Second model
with DimViz() as viz2:
    output2 = model_v2(x)

# Compare
diff = compare_logs(viz1.get_log(), viz2.get_log(), "v1", "v2")
print(diff)
```

##  Use Cases

### 1. Debugging Shape Mismatches

```python
# Find where dimensions go wrong
with DimViz():
    x = torch.randn(32, 3, 224, 224)
    x = conv1(x)  # (32, 64, 112, 112)
    x = conv2(x)  # (32, 128, 56, 56)
    x = x.view(32, -1)  # See the flattened size!
    x = fc(x)  # Does it match?
```

### 2. Understanding Complex Architectures

```python
# Trace transformer attention mechanism
with DimViz(verbose=False):  # Only shape changes
    attention_output = transformer_layer(x)
```

### 3. Optimizing Memory Usage

```python
# Find memory-hungry operations
with DimViz(track_memory=True, filter_ops=['conv2d', 'linear']):
    output = large_model(x)
```

### 4. Teaching & Documentation

```python
# Generate shape flow documentation
@visualize(verbose=True)
def forward_pass(x):
    """Documented forward pass with shape tracking."""
    return model(x)
```

## 🔧 Configuration Options

### `DimViz()`

| Parameter      | Type      | Default | Description                                             |
| -------------- | --------- | ------- | ------------------------------------------------------- |
| `verbose`      | bool      | `True`  | Log all operations (True) or only shape changes (False) |
| `track_memory` | bool      | `False` | Track memory allocation per operation                   |
| `filter_ops`   | List[str] | `None`  | Only track specific operations                          |
| `max_entries`  | int       | `None`  | Limit number of logged operations                       |
| `show_summary` | bool      | `True`  | Display summary statistics                              |

### `@visualize()`

Same parameters as `DimViz()`, used as a decorator:

```python
@visualize(verbose=False, track_memory=True)
def my_function(x):
    return model(x)
```

## 📊 Supported Operations

DimViz translates PyTorch operations to friendly names:

- `aten::mm` → `matmul`
- `aten::addmm` → `linear_proj`
- `aten::conv2d` → `conv2d`
- `aten::bmm` → `batch_matmul`
- `aten::cat` → `concat`
- `aten::sigmoid` → `sigmoid`
- ...and many more!

Operations are automatically detected and logged with human-readable names.

## 🎨 Output Formats

### Terminal Output

**With Rich** (if installed):

- Colored, styled tables
- Better readability
- Automatic terminal adaptation

**Without Rich**:

- ASCII tables via tabulate
- Still clear and readable
- Works everywhere

### Export Formats

**JSON** - Structured data with metadata:

```json
{
  "metadata": {
    "timestamp": "2024-01-29T10:30:00",
    "total_operations": 10
  },
  "log": [
    {
      "step": 1,
      "operation": "linear",
      "input_shape": "(32, 784)",
      "output_shape": "(32, 256)"
    }
  ]
}
```

**CSV** - Spreadsheet-friendly:

```csv
step,operation,input_shape,output_shape
1,linear,"(32, 784)","(32, 256)"
2,relu,"(32, 256)","(32, 256)"
```

**TXT** - Human-readable logs:

```
DimViz Log Export
================================================================================
Step | Operation | Input Shape | Output Shape
1    | linear    | (32, 784)   | (32, 256)
2    | relu      | (32, 256)   | (32, 256)
```

## ⚡ Performance

DimViz uses PyTorch's `TorchDispatchMode` for minimal overhead:

- **Verbose mode**: ~5-10% slowdown (logs everything)
- **Non-verbose mode**: ~2-5% slowdown (logs only shape changes)
- **With filtering**: ~1-3% slowdown (logs specific ops only)

For production code, use `verbose=False` or `filter_ops` to minimize impact.

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
git clone https://github.com/yourusername/dimviz.git
cd dimviz
pip install -e ".[dev]"
pytest tests/
```

### Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=dimviz --cov-report=html
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Built on PyTorch's `TorchDispatchMode` API
- Visualization powered by [Rich](https://github.com/Textualize/rich) (optional)
- Table formatting by [tabulate](https://github.com/astanin/python-tabulate)

## 📧 Contact

- GitHub: [@himalaya-kaushik](https://github.com/yourusername)
- Email: himalaya341@gmail.com

---

**Made with ❤️ for the PyTorch community**

If you find DimViz helpful, please consider giving it a ⭐ on GitHub!
