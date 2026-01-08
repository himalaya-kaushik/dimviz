import torch
import torch.nn as nn
from torchdimviz import visualize

class DeepPyTorchLesson(nn.Module):
    def __init__(self):
        super().__init__()
        # Common scenario: Image -> Flatten -> Dense
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(16 * 28, 20, batch_first=True) # Intentional weird architecture
        self.fc = nn.Linear(20, 10)

    @visualize
    def forward(self, x):
        # 1. Conv Layer
        x = self.conv(x)
        
        # 2. Permute (The invisible shape killer)
        # Change from [Batch, Channel, H, W] -> [Batch, H, Channel*W] for LSTM?
        # Let's say we mess up and just flatten:
        b, c, h, w = x.shape
        # x = x.view(b, h, c * w) 
        x = x.view(b, h, -1)
        
        # 3. LSTM
        x, _ = self.lstm(x)
        
        # 4. Select last step
        x = x[:, -1, :]
        
        # 5. Output
        return self.fc(x)

# Run it
model = DeepPyTorchLesson()
input_data = torch.randn(2, 3, 28, 28)

print("--- Running Model ---")
out = model(input_data)
print("--- Done ---")