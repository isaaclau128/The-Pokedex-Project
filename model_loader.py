"""
Pokemon Model Loader
Loads the trained model without running notebook cells

Use this in your notebook or Python scripts to access the model directly.
"""

import torch
import torch.nn as nn
import os

IMG_SIZE = 128
NUM_CLASSES = 1000
MODEL_PATH = './pokemon_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNClassifier(nn.Module):
    """Matches the model from Pokedex.ipynb"""
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 256)
        self.fc2 = nn.Linear(256, NUM_CLASSES)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def load_model(model_path=MODEL_PATH, device=DEVICE):
    """
    Load trained model from disk
    
    Args:
        model_path: Path to saved model weights
        device: torch device to load onto
        
    Returns:
        model: Loaded CNN model ready for inference
        
    Raises:
        FileNotFoundError: If model doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Please run Pokedex.ipynb to train and save the model first."
        )
    
    model = CNNClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✓ Model loaded from {model_path}")
    return model


def model_exists(model_path=MODEL_PATH):
    """Check if model file exists"""
    return os.path.exists(model_path)


# Quick convenience function
def get_model():
    """Get model - handles errors gracefully"""
    try:
        return load_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    # Test usage
    print("Testing model loader...")
    try:
        model = load_model()
        print("✓ Model loaded successfully!")
        print(f"  Device: {DEVICE}")
        print(f"  Model ready for inference")
    except FileNotFoundError as e:
        print(f"✗ {e}")
