import torch
import torch.nn as nn
from tabulate import tabulate
import functools
import sys

# Global log to store our trace
TRACE_LOG = []

# List of tensor operations we want to spy on
# These are the "invisible" shape killers
TENSOR_OPS = ['view', 'reshape', 'permute', 'transpose', 'flatten', 'squeeze', 'unsqueeze']

# Store original functions so we can restore them later
ORIGINALS = {
    'module_call': nn.Module.__call__,
    'tensor_ops': {op: getattr(torch.Tensor, op) for op in TENSOR_OPS}
}

def _clean_shape(obj):
    """Format shape nicely for the table."""
    if isinstance(obj, torch.Tensor):
        return str(tuple(obj.shape))
    if isinstance(obj, (tuple, list)):
        return " | ".join([_clean_shape(x) for x in obj if isinstance(x, torch.Tensor)])
    return ""

# --- 1. THE LAYER HIJACKER ---
def wrapped_module_call(self, *args, **kwargs):
    # Run the actual layer
    output = ORIGINALS['module_call'](self, *args, **kwargs)
    
    # Log it (only if it's a leaf layer, not a container like Sequential)
    # We filter out containers to keep the log clean
    has_children = len(list(self.children())) > 0
    if not has_children:
        shape_str = _clean_shape(output)
        if shape_str:
            TRACE_LOG.append(["Layer", f"{self.__class__.__name__}", shape_str])
            
    return output

# --- 2. THE TENSOR OP HIJACKER ---
def make_op_wrapper(op_name):
    original_op = ORIGINALS['tensor_ops'][op_name]
    
    @functools.wraps(original_op)
    def wrapped_op(self, *args, **kwargs):
        # Run the actual operation
        output = original_op(self, *args, **kwargs)
        
        # Log it
        shape_str = _clean_shape(output)
        
        # Make the arguments readable (e.g., view(-1, 20))
        arg_str = ", ".join([str(a) for a in args])
        
        TRACE_LOG.append(["Op", f".{op_name}({arg_str})", shape_str])
        return output
        
    return wrapped_op

# --- 3. THE MANAGER ---
class DimensionVisualizer:
    def __init__(self):
        pass

    def __enter__(self):
        # Clear log
        TRACE_LOG.clear()
        
        # 1. Hijack Layers
        nn.Module.__call__ = wrapped_module_call
        
        # 2. Hijack Operations
        for op in TENSOR_OPS:
            setattr(torch.Tensor, op, make_op_wrapper(op))
        
        print("\n[DimVis] 🕵️  Spying on model execution...\n")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # RESTORE EVERYTHING IMMEDIATELY
        # This is critical so we don't break PyTorch for the rest of the script
        nn.Module.__call__ = ORIGINALS['module_call']
        for op in TENSOR_OPS:
            setattr(torch.Tensor, op, ORIGINALS['tensor_ops'][op])

        # Print Report
        if exc_type:
            print(f"\n[DimVis] 💥 CRASH DETECTED: {exc_value}")
            print("Here is the flow just before the crash:\n")
        
        headers = ["Type", "Operation", "Output Shape"]
        print(tabulate(TRACE_LOG, headers=headers, tablefmt="fancy_grid"))
        print("\n[DimVis] Done.")

# --- 4. THE DECORATOR SHORTCUT ---
def visualize(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # We only visualize if we haven't done it yet for this object
        if not getattr(self, "_has_logged_dims", False):
            with DimensionVisualizer():
                result = func(self, *args, **kwargs)
            self._has_logged_dims = True
            return result
        else:
            return func(self, *args, **kwargs)
    return wrapper