import torch
from torch.utils._python_dispatch import TorchDispatchMode
from tabulate import tabulate
import functools

# --- CONFIGURATION ---

FRIENDLY_NAMES = {
    't': 'transpose',
    'mm': 'matmul',
    'addmm': 'linear_proj',         # What nn.Linear uses
    'bmm': 'batch_matmul',
    'select': 'slice_index',        # x[:, -1]
    'unsafe_split': 'split',        # LSTM internal
    'unsafe_view': 'view',
    'unbind': 'unstack',
    'cat': 'concat',
    'mul': 'multiply',
    'div': 'divide',
    'sub': 'subtract',
    'add': 'add',
    'sigmoid': 'sigmoid',
    'tanh': 'tanh',
}

# 2. Noise Gate - Internal ops to ignore (Memory/Size checks)
IGNORED_OPS = {
    'aten::size', 'aten::stride', 'aten::storage_offset', 'aten::is_floating_point',
    'aten::is_complex', 'aten::is_conj', 'aten::numel', 'aten::dim',
    'aten::detach', 'aten::empty', 'aten::empty_like', 'aten::as_strided',
    'aten::_local_scalar_dense'
}

class DimVizTracker(TorchDispatchMode):
    def __init__(self, verbose=True):
        self.log = []
        self.step = 0
        self.verbose = verbose

    def _format_shape(self, obj):
        if isinstance(obj, torch.Tensor):
            return str(tuple(obj.shape))
        if isinstance(obj, (list, tuple)):
            # Recursively format lists of tensors
            shapes = [self._format_shape(x) for x in obj if isinstance(x, (torch.Tensor, list, tuple))]
            if shapes:
                return " | ".join(shapes)
        return ""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None: kwargs = {}
        
        op_name = func._schema.name
        
        # Run execution first (Let PyTorch do the math)
        output = func(*args, **kwargs)

        if op_name not in IGNORED_OPS:
            # 1. Clean the name (remove 'aten::')
            raw_name = op_name.replace("aten::", "")
            # 2. Translate (Get friendly name if exists, else use raw)
            # Remove trailing underscore for in-place ops (add_ -> add) for lookup
            lookup_name = raw_name.rstrip('_')
            friendly_base = FRIENDLY_NAMES.get(lookup_name, raw_name)
            
            # Restore underscore if it was in-place
            if raw_name.endswith('_') and not friendly_base.endswith('_'):
                clean_name = friendly_base + "_"
            else:
                clean_name = friendly_base
            
            # 3. Get Shapes
            in_shape = self._format_shape(args[0]) if args and isinstance(args[0], (torch.Tensor, list, tuple)) else ""
            out_shape = self._format_shape(output)

            # 4. Log Selection Logic
            if in_shape and out_shape:
                # If verbose=True (Default), we show EVERYTHING except ignored ops.
                # If verbose=False, we only show shape changes.
                shape_changed = (in_shape != out_shape)
                
                if self.verbose or shape_changed:
                    self.step += 1
                    self.log.append([self.step, clean_name, in_shape, out_shape])

        return output

class DimViz:
    """
    Usage:
        with DimViz():  <-- Defaults to verbose=True (Show everything)
            model(x)
    """
    def __init__(self, verbose=True):
        self.tracker = DimVizTracker(verbose=verbose)

    def __enter__(self):
        print(f"\n[DimViz] 🟢 Tracking Started...")
        self.tracker.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.tracker.__exit__(exc_type, exc_value, traceback)
        print("[DimViz] 🔴 Tracking Finished.")
        
        if exc_type:
            print(f"\n[DimViz] ⚠️ CRASH DETECTED: {exc_value}")

        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box
            console = Console()
            table = Table(title="Dimension Flow Log", box=box.ROUNDED)
            table.add_column("Step", style="dim", justify="right")
            table.add_column("Operation", style="cyan")
            table.add_column("Input Shape", style="yellow")
            table.add_column("Output Shape", style="bold green")
            for row in self.tracker.log:
                table.add_row(str(row[0]), row[1], row[2], row[3])
            console.print(table)
        except ImportError:
            print(tabulate(self.tracker.log, headers=["Step", "Op", "Input", "Output"], tablefmt="fancy_grid"))

def visualize(verbose=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with DimViz(verbose=verbose):
                return func(*args, **kwargs)
        return wrapper
    return decorator