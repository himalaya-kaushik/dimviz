import torch
import torch.nn as nn
import torch.fx
from tabulate import tabulate
import functools

# ==========================================
# PART 1: THE DEBUGGER TOOL (Embedded here)
# ==========================================

def _get_shape_str(obj):
    """
    Robustly extract shape info.
    If it's not a tensor, it tells us WHAT it is (e.g., 'int', 'NoneType').
    """
    if isinstance(obj, torch.Tensor):
        return f"Tensor{tuple(obj.shape)}"
    
    if isinstance(obj, (tuple, list)):
        return " | ".join([_get_shape_str(x) for x in obj])
    
    # Debugging: If it's not a tensor, print the Type so we know why it failed
    return f"<{type(obj).__name__}>"

def _run_debug_profile(model, *args):
    print(f"\n[DimVis] 🔍 Analyzing {model.__class__.__name__}...")
    
    try:
        # 1. Trace the graph
        traced = torch.fx.symbolic_trace(model)
        
        # 2. Setup the Spy
        class ShapeSpy(torch.fx.Interpreter):
            def run_node(self, n):
                # Run the node
                result = super().run_node(n)
                
                # Inspect the output
                shape_info = _get_shape_str(result)
                
                # Clean up names for the report
                op_name = str(n.target).replace("built-in method ", "")
                if n.op == 'placeholder': op_name = "Input"
                elif n.op == 'call_module': op_name = f"Layer: {n.target}"
                elif n.op == 'call_method': op_name = f".{n.target}()"
                elif n.op == 'output': op_name = "Return"
                
                # Store row (skip internal getattr access)
                if n.op != 'get_attr':
                    self.log.append([n.name, n.op, op_name, shape_info])
                return result

        # 3. Initialize and Run
        spy = ShapeSpy(traced)
        spy.log = [] # Custom list to store our rows
        spy.run(*args)
        
        # 4. Print
        headers = ["Node", "Type", "Operation", "Output Shape / Type"]
        print(tabulate(spy.log, headers=headers, tablefmt="fancy_grid"))
        
    except Exception as e:
        print(f"\n[DimVis] CRASHED: {e}")
        import traceback
        traceback.print_exc()

def visualize(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "_has_run", False):
            if isinstance(self, nn.Module):
                # Detach inputs
                d_args = [a.detach() if isinstance(a, torch.Tensor) else a for a in args]
                _run_debug_profile(self, *d_args)
                self._has_run = True
        return func(self, *args, **kwargs)
    return wrapper

# ==========================================
# PART 2: THE TEST MODEL (Linear, No Loops)
# ==========================================

class SimpleTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16*28*28, 10)

    @visualize
    def forward(self, x):
        # 1. Conv
        x = self.conv(x)
        
        # 2. View (Flatten)
        # Using hardcoded -1 to avoid symbolic math issues for this test
        x = x.view(x.size(0), -1)
        
        # 3. Linear
        x = self.fc(x)
        return x

# ==========================================
# PART 3: EXECUTION
# ==========================================

print("--- STARTING SINGLE FILE TEST ---")
model = SimpleTest()
input_data = torch.randn(2, 3, 28, 28) # Batch 2, 3 Channels, 28x28
out = model(input_data)
print("--- TEST FINISHED ---")