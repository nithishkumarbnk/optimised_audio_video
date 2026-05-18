"""Inspect PyTorch model outputs to understand class mappings."""
import torch
import numpy as np
import cv2


def inspect(name: str, model_path: str) -> None:
    print(f"\n=== {name} ===")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    # Create a dummy 640x640 RGB frame
    dummy = np.zeros((640, 640, 3), dtype=np.float32)
    tensor = torch.from_numpy(dummy.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    raw = output[0].cpu().float().numpy()  # (channels, 8400)
    print(f"  output shape: {raw.shape}")
    preds = raw.T  # (8400, channels)
    class_scores = preds[:, 4:]
    print(f"  class_scores shape: {class_scores.shape}")
    for i in range(class_scores.shape[1]):
        mx = class_scores[:, i].max()
        top_idx = class_scores[:, i].argmax()
        print(f"  class {i}: max_conf={mx:.4f} (anchor {top_idx})")


inspect("headset_model", "app/models/headset_model.pt")
inspect("proctoring", "app/models/proctoring.pt")
