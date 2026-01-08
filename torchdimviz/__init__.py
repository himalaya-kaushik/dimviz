import torch
import torch.nn as nn
import torch.fx
from tabulate import tabulate
import functools

# --- OPTIONAL: Pretty Printing with Rich ---
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# --- 1. THE CLEANER (Formats output) ---
def _clean_shape(obj):
    """
    Extracts shape from an object (Tensor, Tuple, List) without crashing.
    """
    if isinstance(obj, torch.Tensor):
        return str(tuple(obj.shape))
    
    if isinstance(obj, (tuple, list)):
        # Recursively clean items inside tuple/list
        return " | ".join([_clean_shape(x) for x in obj])
        
    return "N/A" # For non-tensor outputs (like None or scalar)

# --- 2. THE NEW ENGINE (No SymPy!) ---
def _run_profile(model, *args):
    """
    Uses a custom Interpreter to run the graph line-by-line.
    This BYPASSES the 'ShapeProp' / SymPy crash entirely.
    """
    try:
        # A. TRACE
        traced_model = torch.fx.symbolic_trace(model)
        
        # B. PREPARE TABLE
        table_rows = []
        
        # C. DEFINE THE SPY
        # We subclass Interpreter to inject our logging logic
        class ShapeSpy(torch.fx.Interpreter):
            def run_node(self, n):
                # 1. Run the actual operation (Execute the math)
                result = super().run_node(n)
                
                # 2. Spy on the result (Get the shape)
                shape_str = _clean_shape(result)
                
                # 3. Log it
                # Clean up target name
                target = str(n.target).replace("built-in method ", "").replace("built-in function ", "")
                if n.op == 'call_method': target = f".{target}()"
                elif n.op == 'call_module': target = f"Layer: {target}"
                elif n.op == 'placeholder': target = "Input"
                elif n.op == 'output': target = "Return"
                
                # Only log relevant nodes (skip boring GetAttr access)
                if n.op != 'get_attr':
                    table_rows.append([n.name, n.op, target, shape_str])
                
                return result

        # D. EXECUTE (This runs the model with REAL data)
        print(f"\n[DimVis] 🔍 Dimension Flow: {model.__class__.__name__}")
        ShapeSpy(traced_model).run(*args)
        
        # E. PRINT REPORT
        if HAS_RICH:
            console = Console()
            table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
            table.add_column("Node", style="dim cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Operation", style="green")
            table.add_column("Output Shape", style="bold white")
            for row in table_rows: table.add_row(*row)
            console.print(table)
        else:
            headers = ["Node", "Type", "Op", "Shape"]
            print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))
            
    except Exception as e:
        print(f"\n[DimVis] ⚠️ Visualization Error: {e}")
        print("[DimVis] Continuing execution...")

# --- 3. THE DECORATOR ---
def visualize(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "_has_visualized_dims", False):
            if isinstance(self, nn.Module):
                # Detach args to be safe
                detached_args = [a.detach() if isinstance(a, torch.Tensor) else a for a in args]
                
                # Run the new engine
                _run_profile(self, *detached_args)
                
                self._has_visualized_dims = True
        return func(self, *args, **kwargs)
    return wrapper