"""
Fine-tuning script for domain adaptation.
Takes pre-trained model and fine-tunes on collected camera frames to close the domain gap.

Usage:
    python finetune_on_camera_frames.py

Expected result: Model accuracy on camera frames improves from ~0.7 to >0.85
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from datetime import datetime

# ===================== Configuration =====================
CAMERA_FRAMES_DIR = "./camera_frames_for_finetuning"
METADATA_FILE = os.path.join(CAMERA_FRAMES_DIR, "metadata.json")
PRETRAINED_MODEL_PATH = "./pokemon_model.pth"
CLASSES_JSON = "./classes.json"
OUTPUT_MODEL_PATH = "./pokemon_model_finetuned.pth"

IMG_SIZE = 224
BATCH_SIZE = 16  # Small batch for domain adaptation
LEARNING_RATE = 1e-4  # Low LR to avoid catastrophic forgetting
NUM_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ===================== Load Classes =====================
if not os.path.exists(CLASSES_JSON):
    print(f"❌ {CLASSES_JSON} not found")
    exit(1)

with open(CLASSES_JSON, 'r') as f:
    class_names = json.load(f)

print(f"✓ Loaded {len(class_names)} Pokemon classes")

# ===================== Camera Dataset =====================
class CameraFramesDataset(Dataset):
    def __init__(self, metadata_file, img_dir, class_names, transform=None):
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        
        # Filter out any labels not in class_names
        self.metadata = [
            item for item in self.metadata
            if item['label'] in self.class_to_idx
        ]
        
        if not self.metadata:
            raise ValueError(f"No valid labels found in metadata. Check class names in {CLASSES_JSON}")
        
        print(f"✓ Loaded {len(self.metadata)} camera frames")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = os.path.join(self.img_dir, item['image_path'])
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[item['label']]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ===================== Model Setup =====================
def create_resnet50_model(num_classes=1000):
    """Create ResNet50 model."""
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# Load pre-trained model
if not os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"❌ Pre-trained model not found at {PRETRAINED_MODEL_PATH}")
    print(f"Please train the main model first: python Pokedex_ResNet50.ipynb")
    exit(1)

model = create_resnet50_model(num_classes=len(class_names)).to(DEVICE)
state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
print(f"✓ Loaded pre-trained model from {PRETRAINED_MODEL_PATH}")

# ===================== Data Transforms =====================
# Use light augmentation for fine-tuning (preserve camera domain)
train_transform = transforms.Compose([
    transforms.RandomRotation(5),  # Light rotation
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Light color jitter
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# ===================== Load Camera Frames =====================
if not os.path.exists(METADATA_FILE):
    print(f"❌ Metadata not found at {METADATA_FILE}")
    print(f"Run collect_camera_frames.py first to collect labeled frames")
    exit(1)

dataset = CameraFramesDataset(METADATA_FILE, CAMERA_FRAMES_DIR, class_names, train_transform)
num_frames = len(dataset)

if num_frames < 20:
    print(f"⚠ Warning: Only {num_frames} frames. Recommend at least 100 for good fine-tuning.")

# Split into train/val
train_size = int(0.8 * num_frames)
val_size = num_frames - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# Update val_dataset transform
val_dataset.dataset.transform = eval_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✓ Created DataLoaders: {train_size} train, {val_size} val")

# ===================== Training Setup =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
)

# ===================== Training Function =====================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# ===================== Fine-tuning =====================
print("\n" + "="*70)
print(f"Starting fine-tuning on {num_frames} camera frames")
print("="*70)

best_val_loss = float('inf')
best_state_dict = None

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion, DEVICE)
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"  Train: Loss={train_loss:.4f}, Accuracy={train_acc:.2%}")
    print(f"  Val:   Loss={val_loss:.4f}, Accuracy={val_acc:.2%}")

# ===================== Save Fine-tuned Model =====================
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

# Backup old model
if os.path.exists(PRETRAINED_MODEL_PATH):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"pokemon_model_pretrain_{timestamp}.pth"
    import shutil
    shutil.copy2(PRETRAINED_MODEL_PATH, backup_path)
    print(f"✓ Backed up original model to {backup_path}")

# Save fine-tuned model as the new main model
torch.save(model.state_dict(), PRETRAINED_MODEL_PATH)
print(f"✓ Saved fine-tuned model to {PRETRAINED_MODEL_PATH}")

# Also save as separate version
torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
print(f"✓ Also saved to {OUTPUT_MODEL_PATH}")

print("\n" + "="*70)
print("✓ Fine-tuning complete!")
print("="*70)
print("\nNext steps:")
print("  1. Test camera detection: python camera_pokemon_recognizer.py")
print("  2. If still poor, collect more frames and re-run this script")
print("  3. To collect more frames: python collect_camera_frames.py")
