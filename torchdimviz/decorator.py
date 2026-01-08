# import torch
# import torch.nn as nn
# import functools

# # --- PART 1: The "Baby" Visualizer (Decorator) ---
# def debug_shapes(func):
#     """
#     A simple decorator that prints the name of the function 
#     and the shape of the FIRST argument (assuming it's a tensor).
#     """
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         print(f"\n[Visualizer] Entering: {func.__name__}")
        
#         # 1. Inspect the Input
#         # args[0] is usually 'self' for class methods, so args[1] is the input 'x'
#         # BUT if it's a standalone function, args[0] is 'x'.
#         # Let's try to find the first Tensor in the inputs.
        
#         found_input = False
#         for i, arg in enumerate(args):
#             if isinstance(arg, torch.Tensor):
#                 print(f"  > Input found at arg[{i}] | Shape: {tuple(arg.shape)}")
#                 found_input = True
#                 break
        
#         if not found_input:
#             print("  > No tensor input found.")

#         # 2. Run the actual function
#         try:
#             result = func(*args, **kwargs)
#         except Exception as e:
#             print(f"  > CRASHED inside {func.__name__}!")
#             raise e # Re-raise the error so we see the stack trace
        
#         # 3. Inspect the Output
#         if isinstance(result, torch.Tensor):
#              print(f"  > Output Shape: {tuple(result.shape)}")
#         else:
#              print(f"  > Output is not a tensor (Type: {type(result)})")
             
#         print(f"[Visualizer] Exiting: {func.__name__}\n")
#         return result
#     return wrapper

# # --- PART 2: The User's Code (Your "Student" Scenario) ---

# class SimpleLSTM(nn.Module):
#     def __init__(self, input_sz, hidden_sz, out_sz):
#         super().__init__()
#         self.lstm = nn.LSTM(input_sz, hidden_sz, batch_first=True)
#         self.fc = nn.Linear(hidden_sz, out_sz)

#     @debug_shapes  
#     def forward(self, x):
#         # x shape: [Batch, Seq, Feature]
#         out, _ = self.lstm(x)
        
#         # Grab the last token.
#         # This is where students often get confused: 
#         # "Did I keep the batch dim? Is it 2D or 3D now?"
#         last_hidden = out[:, -1, :] 
        
#         return self.fc(last_hidden)

# # --- PART 3: Execution ---

# # Setup
# model = SimpleLSTM(input_sz=10, hidden_sz=20, out_sz=5)
# dummy_input = torch.randn(32, 15, 10) # [Batch=32, Seq=15, Feat=10]

# print("--- Starting Run ---")
# output = model(dummy_input)
# print("--- Finished Run ---")