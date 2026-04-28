#!/usr/bin/env python3
"""
Model Loading Examples
Different ways to load and use your trained Pokemon model
"""

import torch
import torch.nn as nn
import os


# ============================================
# EXAMPLE 1: Using model_loader.py
# ============================================
def example_1_use_model_loader():
    """Cleanest approach - recommended for most use cases"""
    print("\n" + "="*50)
    print("EXAMPLE 1: Using model_loader.py")
    print("="*50)
    
    try:
        from model_loader import load_model
        print("✓ Importing model_loader...")
        
        model = load_model()
        print(f"✓ Model loaded: {type(model).__name__}")
        print(f"✓ Ready for inference!")
        
    except ImportError:
        print("✗ model_loader not found in current directory")
    except FileNotFoundError as e:
        print(f"✗ {e}")


# ============================================
# EXAMPLE 2: Direct PyTorch Loading
# ============================================
def example_2_direct_loading():
    """Standalone approach - no dependencies"""
    print("\n" + "="*50)
    print("EXAMPLE 2: Direct PyTorch Loading")
    print("="*50)
    
    # Define model locally
    class CNNClassifier(nn.Module):
        def __init__(self):
            super(CNNClassifier, self).__init__()
            IMG_SIZE = 128
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(128 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 256)
            self.fc2 = nn.Linear(256, 1000)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './pokemon_model.pth'
    
    try:
        print(f"Loading from: {model_path}")
        model = CNNClassifier().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✓ Model loaded successfully")
        print(f"✓ Device: {device}")
        
    except FileNotFoundError:
        print(f"✗ Model file not found at {model_path}")


# ============================================
# EXAMPLE 3: Conditional Loading (Skip if trained)
# ============================================
def example_3_conditional_loading():
    """Only load if model exists, show status"""
    print("\n" + "="*50)
    print("EXAMPLE 3: Conditional Loading")
    print("="*50)
    
    model_path = './pokemon_model.pth'
    
    if os.path.exists(model_path):
        print(f"✓ Model found at {model_path}")
        
        try:
            from model_loader import load_model
            model = load_model()
            
            file_size = os.path.getsize(model_path) / 1024 / 1024
            print(f"✓ Model size: {file_size:.2f} MB")
            print(f"✓ Model is ready for inference")
            return model
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return None
    else:
        print(f"✗ Model not found at {model_path}")
        print(f"  Please run Pokedex.ipynb to train and save the model first")
        return None


# ============================================
# EXAMPLE 4: Using in a Prediction Pipeline
# ============================================
def example_4_prediction_pipeline():
    """Complete example with preprocessing and prediction"""
    print("\n" + "="*50)
    print("EXAMPLE 4: Prediction Pipeline")
    print("="*50)
    
    try:
        from model_loader import load_model
        import numpy as np
        from torchvision import transforms
        from PIL import Image
        
        print("Loading model...")
        model = load_model()
        
        print("Setting up preprocessing...")
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        print("✓ Ready for predictions!")
        print("\nUsage:")
        print("  1. Load image: img = Image.open('pokemon.jpg')")
        print("  2. Preprocess: tensor = transform(img).unsqueeze(0)")
        print("  3. Predict: output = model(tensor)")
        print("  4. Get results: probs = torch.softmax(output, dim=1)")
        
    except Exception as e:
        print(f"Error: {e}")


# ============================================
# EXAMPLE 5: Model Information
# ============================================
def example_5_model_info():
    """Show model architecture and details"""
    print("\n" + "="*50)
    print("EXAMPLE 5: Model Information")
    print("="*50)
    
    try:
        from model_loader import load_model
        
        model = load_model()
        
        print("\nModel Architecture:")
        print(model)
        
        print("\nModel Parameters:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        print("\nInput/Output:")
        print(f"  Input shape: (batch, 3, 128, 128)")
        print(f"  Output shape: (batch, 1000)")
        
    except Exception as e:
        print(f"Error: {e}")


# ============================================
# MAIN
# ============================================
def main():
    print("\n" + "="*50)
    print("Pokemon Model Loading Examples")
    print("="*50)
    
    # Run all examples
    example_1_use_model_loader()
    example_2_direct_loading()
    example_3_conditional_loading()
    example_4_prediction_pipeline()
    example_5_model_info()
    
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print("\n✓ All examples completed!")
    print("\nRecommended approach:")
    print("  Use EXAMPLE 1 (model_loader.py) in your projects")
    print("\nFor more info, see LOAD_MODEL_GUIDE.md")


if __name__ == "__main__":
    main()
