from pathlib import Path
import os
import shutil
import ssl
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
import torch.nn as nn
from torchvision import models

try:
    import certifi
except ImportError:
    certifi = None


GEN1_DIR = Path(__file__).resolve().parent
MODEL_PATH = GEN1_DIR / 'pokemon_model_gen1_resnet50.pth'
CLASSES_PATH = GEN1_DIR / 'gen1_classes.json'
IMG_SIZE = 224


def _configure_ssl():
    if certifi is None:
        return None

    ca_bundle = certifi.where()
    os.environ['SSL_CERT_FILE'] = ca_bundle
    os.environ['REQUESTS_CA_BUNDLE'] = ca_bundle
    os.environ['CURL_CA_BUNDLE'] = ca_bundle
    return ca_bundle


CA_BUNDLE = _configure_ssl()


def _download_weight_file(url):
    checkpoint_dir = Path(torch.hub.get_dir()) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / Path(urlparse(url).path).name

    if checkpoint_path.exists():
        return checkpoint_path

    ssl_context = ssl.create_default_context(cafile=CA_BUNDLE) if CA_BUNDLE is not None else None
    with urlopen(url, context=ssl_context) as response, checkpoint_path.open('wb') as file_handle:
        shutil.copyfileobj(response, file_handle)

    return checkpoint_path


def _create_resnet50_model(num_classes, use_pretrained=False):
    model = models.resnet50(weights=None)

    if use_pretrained:
        weights = models.ResNet50_Weights.DEFAULT
        checkpoint_path = _download_weight_file(weights.url)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_class_names(classes_path=CLASSES_PATH):
    import json

    if not Path(classes_path).exists():
        raise FileNotFoundError(f'Class file not found: {classes_path}')

    with open(classes_path, 'r') as file_handle:
        return json.load(file_handle)


def model_exists(model_path=MODEL_PATH):
    return Path(model_path).exists()


def load_model(model_path=MODEL_PATH, classes_path=CLASSES_PATH, device=None):
    class_names = get_class_names(classes_path)
    device = device or get_device()

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f'Model not found at {model_path}. Train the notebook in gen1/gen1.ipynb first.'
        )

    model = _create_resnet50_model(len(class_names), use_pretrained=False).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_names, device