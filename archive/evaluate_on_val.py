"""
Evaluate the saved model on the validation set and report accuracy
and a short summary of prediction confidences to detect overconfidence.

Usage:
  python evaluate_on_val.py
"""
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model_loader import load_model

IMG_SIZE = 128
BATCH_SIZE = 32
VAL_DIR = './pokemon-dataset-1000/pokemon-dataset-1000/val'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_val_loader(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Validation folder not found: {path}")
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return loader, dataset.classes


def evaluate():
    model = load_model(device=DEVICE)
    if model is None:
        print('Failed to load model')
        return

    val_loader, classes = load_val_loader(VAL_DIR)
    model.to(DEVICE)
    model.eval()

    total = 0
    correct = 0
    confidences = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            top_probs, preds = probs.max(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            confidences.extend(top_probs.cpu().tolist())

    acc = correct / total if total > 0 else 0.0
    print(f"Validation samples: {total}")
    print(f"Top-1 Accuracy on validation: {acc:.4f} ({correct}/{total})")

    # Confidence stats
    import numpy as np
    conf = np.array(confidences)
    print("Confidence stats (top predicted class):")
    print(f"  mean: {conf.mean():.4f}")
    print(f"  median: {np.median(conf):.4f}")
    print(f"  std: {conf.std():.4f}")
    print(f"  <0.5: {(conf < 0.5).sum()}  |  <0.2: {(conf < 0.2).sum()}")

    # Show class-wise accuracy for first 20 classes (compact)
    try:
        from collections import defaultdict
        per_class_total = defaultdict(int)
        per_class_correct = defaultdict(int)
        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                    per_class_total[t] += 1
                    if t == p:
                        per_class_correct[t] += 1
        print("\nSample per-class accuracies (first 20 classes):")
        for i, cname in enumerate(classes[:20]):
            tot = per_class_total.get(i, 0)
            corr = per_class_correct.get(i, 0)
            acc_cls = corr / tot if tot > 0 else 0.0
            print(f"  {i:03d} {cname}: {acc_cls:.3f} ({corr}/{tot})")
    except Exception as e:
        print('Failed computing per-class accuracies:', e)


if __name__ == '__main__':
    evaluate()
