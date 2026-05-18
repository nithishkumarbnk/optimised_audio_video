"""Read metadata from PyTorch .pt model files."""
import torch

for name, path in [('headset', 'app/models/headset_model.pt'), ('proctor', 'app/models/proctoring.pt')]:
    model = torch.load(path, map_location='cpu', weights_only=False)
    model.eval()
    print(f"{name}: {type(model).__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {total_params:,}")
