"""
Fit a scalar temperature on the validation set to calibrate model confidence.

This does not change top-1 accuracy directly, but it usually reduces
overconfident probabilities when the model is wrong.

Usage:
  python calibrate_temperature.py
"""

import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_loader import load_model

IMG_SIZE = 128
VAL_DIR = './pokemon-dataset-1000/pokemon-dataset-1000/val'
TEMPERATURE_PATH = './temperature.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature.clamp_min(1e-6)


def load_val_data(val_dir):
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f'Validation folder not found: {val_dir}')

    dataset = datasets.ImageFolder(val_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    return loader, dataset.classes


def collect_logits(model, loader, device):
    logits_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            logits_list.append(outputs.detach().cpu())
            labels_list.append(labels.cpu())

    return torch.cat(logits_list), torch.cat(labels_list)


def fit_temperature(logits, labels):
    scaler = TemperatureScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)

    logits = logits.clone().detach()
    labels = labels.clone().detach()

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(scaler.temperature.item())
    return max(temperature, 1e-6)


def save_temperature(temperature, path):
    with open(path, 'w') as f:
        json.dump({'temperature': temperature}, f, indent=2)


def main():
    print('=' * 50)
    print('Pokemon Temperature Calibration')
    print('=' * 50)

    model = load_model(device=DEVICE)
    if model is None:
        raise SystemExit(1)

    loader, classes = load_val_data(VAL_DIR)
    print(f'Validation samples: {len(loader.dataset)}')
    print(f'Validation classes: {len(classes)}')

    logits, labels = collect_logits(model, loader, DEVICE)
    base_nll = nn.CrossEntropyLoss()(logits, labels).item()
    temperature = fit_temperature(logits, labels)
    calibrated_nll = nn.CrossEntropyLoss()(logits / temperature, labels).item()

    save_temperature(temperature, TEMPERATURE_PATH)

    print(f'Base NLL: {base_nll:.4f}')
    print(f'Fitted temperature: {temperature:.4f}')
    print(f'Calibrated NLL: {calibrated_nll:.4f}')
    print(f'Saved temperature to {TEMPERATURE_PATH}')


if __name__ == '__main__':
    main()