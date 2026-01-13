import torch
import torch.nn as nn
import math
from torchdimviz import DimViz

class ComplexHybridModel(nn.Module):
    def __init__(self, img_size=64, patch_size=8, embed_dim=128, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # 1. CNN Feature Extractor (The "Stem")
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) # Reduces 64x64 -> 32x32
        )

        # 2. Projection to Patches (Linear Embedding)
        # We project flattened patches into the embedding dimension
        self.projection = nn.Linear(32 * (patch_size//2) * (patch_size//2), embed_dim)

        # 3. Positional Encoding (Learnable Parameter)
        # We add this manually to test arithmetic logging
        self.pos_embed = nn.Parameter(torch.randn(1, (32//(patch_size//2))**2, embed_dim))

        # 4. Transformer Encoder Layer
        self.transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        
        # 5. Output Head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # --- STAGE 1: Image Processing (4D) ---
        # Input: [Batch, 3, 64, 64]
        x = self.stem(x)
        # Output: [Batch, 32, 32, 32] (Pooled)

        # --- STAGE 2: The "Dimension Hell" (Reshaping to Sequence) ---
        B, C, H, W = x.shape
        
        # Manually unfold patches. 
        # This is usually where people get confused.
        # We want: [Batch, Num_Patches, Flattened_Pixels]
        # First, view as patches
        p = self.patch_size // 2 # Adjusted for pooling
        x = x.view(B, C, H // p, p, W // p, p)
        
        # Permute to group patches together
        x = x.permute(0, 2, 4, 1, 3, 5)
        
        # Flatten the patches
        x = x.reshape(B, (H // p) * (W // p), -1) 
        
        # --- STAGE 3: Sequence Modeling (3D) ---
        # Project to embedding dim
        x = self.projection(x)
        
        # Arithmetic: Add Positional Embeddings
        # (DimViz should catch this 'add' operation)
        x = x + self.pos_embed

        # Run Transformer
        # (DimViz should catch the internal matmuls and layer norms)
        x = self.transformer(x)

        # --- STAGE 4: Classification (2D) ---
        # Global Average Pooling (Reduce Sequence Length to 1)
        x = x.mean(dim=1) 
        
        # Final Linear
        x = self.head(x)
        
        return x

# --- EXECUTION ---

# Create model
model = ComplexHybridModel()
input_data = torch.randn(2, 3, 64, 64) # Batch Size 2

print(f"\n--- COMPLEX MODEL STRUCTURE ---")
print(f"1. CNN Stem (4D)")
print(f"2. Patch Flattening & Permutation (Complex View/Reshape)")
print(f"3. Transformer Self-Attention (3D)")
print(f"4. Global Average Pooling (Reduction)")
print(f"-------------------------------\n")

print("--- Running DimViz (Verbose=True) ---")
# Using verbose=True to see the internal Transformer matmuls and the manual 'add'
with DimViz(verbose=True):
    output = model(input_data)