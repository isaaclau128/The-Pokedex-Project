"""
Diagnostic script to test camera preprocessing on validation images.
Run this to verify camera inference matches validation dataset performance.
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import json
import pandas as pd

# ===================== Configuration =====================
IMG_SIZE = 224  # Must match camera_pokemon_recognizer.py and notebook
MODEL_PATH = "./pokemon_model.pth"
METADATA_PATH = "./pokemon-dataset-1000/metadata.csv"
CLASSES_PATH = "./classes.json"
IMAGE_BASE_DIR = "./pokemon-dataset-1000/pokemon-dataset-1000"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===================== Transforms =====================
camera_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# ===================== Load Model =====================
def create_resnet50_model(num_classes=1000):
    from torchvision import models
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    exit(1)

# Load classes
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"✓ Loaded {len(class_names)} classes from {CLASSES_PATH}")
else:
    print(f"❌ Classes file not found at {CLASSES_PATH}")
    exit(1)

# Load model
model = create_resnet50_model(num_classes=len(class_names)).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"✓ Model loaded from {MODEL_PATH}")

# ===================== Load Sample Validation Images =====================
if not os.path.exists(METADATA_PATH):
    print(f"❌ Metadata not found at {METADATA_PATH}")
    exit(1)

df = pd.read_csv(METADATA_PATH)
print(f"✓ Loaded metadata with {len(df)} images")

# Sample 10 random validation images
sample_indices = np.random.choice(len(df), size=min(10, len(df)), replace=False)
correct = 0
total = 0

print("\n" + "="*80)
print("Testing camera preprocessing on sample validation images:")
print("="*80)

for idx in sample_indices:
    row = df.iloc[idx]
    rel_path = str(row['image_path'])
    img_path = os.path.join(IMAGE_BASE_DIR, rel_path)
    true_label = str(row['label'])
    
    if not os.path.exists(img_path):
        print(f"⚠ Image not found: {img_path}")
        continue
    
    try:
        # Load and preprocess
        img = Image.open(img_path).convert('RGB')
        img_tensor = camera_transform(img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probabilities[0], 0)
            pred_label = class_names[int(pred_idx.cpu().numpy())]
        
        is_correct = pred_label == true_label
        total += 1
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "❌"
        print(f"{status} True: {true_label:20s} | Pred: {pred_label:20s} | Conf: {float(confidence):.3f}")
        
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")

print("="*80)
if total > 0:
    accuracy = correct / total
    print(f"\nCamera Preprocessing Test Accuracy: {accuracy:.1%} ({correct}/{total})")
    if accuracy >= 0.6:
        print("✓ Camera preprocessing looks good!")
    else:
        print("⚠ Accuracy is lower than expected. This may indicate:")
        print("  - Input size mismatch (verify IMG_SIZE=224)")
        print("  - Preprocessing mismatch (verify transforms)")
        print("  - Poor model quality (retrain with IMG_SIZE=224)")
else:
    print("❌ No images tested")
