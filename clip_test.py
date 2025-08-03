import clip
import torch

# Use D: drive as custom download root
download_path = "D:/clip_models"

print(f"Using model cache at: {download_path}")

try:
    model, preprocess = clip.load("ViT-B/32", device="cpu", download_root=download_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading CLIP model:")
    print(str(e))
