# import torch
# import torch.nn as nn
# import torch.fx
# from tabulate import tabulate # You might need: pip install tabulate

# # --- PART 1: The Core Logic (The "Trace Reporter") ---
# import torch
# import torch.nn as nn
# import torch.fx
# from torch.fx.passes.shape_prop import ShapeProp
# from tabulate import tabulate

# def profile_shapes(model, input_data):
#     # 1. TRACE
#     tracer = torch.fx.symbolic_trace(model)
    
#     # 2. RUN (Propagate Shapes)
#     ShapeProp(tracer).propagate(input_data)
    
#     # --- HELPER: Clean up the massive TensorMetadata mess ---
#     def clean_shape(meta):
#         # Case 1: It's a simple Tensor or TensorMetadata (has .shape)
#         if hasattr(meta, 'shape'):
#             return str(tuple(meta.shape))
        
#         # Case 2: It's a Tuple/List (like LSTM output: (out, (h, c)))
#         # We recursively clean the items inside
#         if isinstance(meta, (tuple, list)):
#             return str([clean_shape(m) for m in meta])
            
#         # Case 3: Fallback
#         return str(meta)

#     # 3. REPORT
#     headers = ["Node Name", "Op Type", "Target", "Output Shape"]
#     table_data = []
    
#     print("\n=== DIMENSION X-RAY REPORT ===")
    
#     for node in tracer.graph.nodes:
#         # Get metadata
#         meta = node.meta.get('tensor_meta', "N/A")
        
#         # CLEAN THE SHAPE (The fix)
#         if meta == "N/A":
#             shape_info = "N/A"
#         else:
#             shape_info = clean_shape(meta)

#         # Clean the Target name for readability
#         target = str(node.target).replace("built-in method ", "").replace("built-in function ", "")
#         if node.op == 'call_method':
#             target = f"Tensor.{node.target}()"
            
#         table_data.append([node.name, node.op, target, shape_info])

#     # Using "grid" format handles long text better than "simple_grid"
#     print(tabulate(table_data, headers=headers, tablefmt="grid"))
#     print("==============================\n")

# # --- PART 2: The "Problematic" Student Code ---

# class ConfusingLSTM(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lstm = nn.LSTM(10, 20, batch_first=True)
#         self.fc = nn.Linear(20, 5)

#     def forward(self, x):
#         # x: [Batch, Seq, Feat]
#         x, _ = self.lstm(x) 
        
#         # The hidden "invisible" operations
#         x = x[:, -1, :]  # Slice: Take last token
#         x = x.view(-1, 20) # Redundant view, but common habit
        
#         return self.fc(x)

# # --- PART 3: Execution ---

# model = ConfusingLSTM()
# dummy_input = torch.randn(32, 15, 10) # [Batch, Seq, Feat]

# # Run our tool instead of just calling model(dummy_input)
# profile_shapes(model, dummy_input)