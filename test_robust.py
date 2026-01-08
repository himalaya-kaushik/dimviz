import torch
import torch.nn as nn
from robust_viz import visualize

class DeepPyTorchLesson(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.lstm = nn.LSTM(16 * 28, 20, batch_first=True)
        self.fc = nn.Linear(20, 10)

    @visualize
    def forward(self, x):
        # 1. Conv Layer
        x = self.conv(x)
        
        # 2. Reshape (The logic that confused FX)
        b, c, h, w = x.shape
        # This dynamic math used to crash FX/SymPy. 
        # With Monkey Patching, it works perfectly because it's just Python.
        x = x.view(b, h, c * w) 
        
        # 3. LSTM
        x, _ = self.lstm(x)
        
        # 4. Slicing (Note: Slicing __getitem__ is hard to patch safely, 
        # so we rely on the next layer to show the shape change)
        x = x[:, -1, :]
        
        # 5. Output
        return self.fc(x)

# Run it
model = DeepPyTorchLesson()
# Using Batch Size 2 to prove the math works
input_data = torch.randn(2, 3, 28, 28) 

print("--- Running Model ---")
out = model(input_data)