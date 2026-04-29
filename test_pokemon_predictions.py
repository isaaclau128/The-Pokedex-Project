import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
import numpy as np

# Model configuration
IMG_SIZE = 128
NUM_CLASSES = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture
class CNNClassifier(nn.Module):
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

# Evaluation transform (same as training)
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model
print("Loading trained model...")
model = CNNClassifier().to(device)
model.load_state_dict(torch.load('pokemon_model.pth', map_location=device))
model.eval()
print("✓ Model loaded successfully\n")

# Load class names
with open('classes.json', 'r') as f:
    class_names = json.load(f)

def predict_pokemon(image_path, top_k=5):
    """Predict Pokemon class from image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = eval_transform(image).unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        # Display results
        print(f"Image: {os.path.basename(image_path)}")
        print(f"{'Rank':<6} {'Pokemon':<25} {'Confidence':<12}")
        print("-" * 45)
        
        for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
            pokemon_name = class_names[idx.item()]
            confidence = prob.item() * 100
            print(f"{i+1:<6} {pokemon_name:<25} {confidence:>6.2f}%")
        
        print()
        return class_names[top_indices[0][0].item()], top_probs[0][0].item()
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}\n")
        return None, None

# Test on dataset Pokemon (from the dataset folder)
print("=" * 50)
print("Testing on Pokémon from the dataset:")
print("=" * 50)
print()

# Test a few Pokemon from the dataset if available
dataset_path = './pokemon-dataset-1000/pokemon-dataset-1000/dataset'
test_pokemon = ['clefable', 'ho-oh', 'eevee']

for pokemon in test_pokemon:
    pokemon_dir = os.path.join(dataset_path, pokemon)
    if os.path.exists(pokemon_dir):
        # Get first image from pokemon folder
        images = [f for f in os.listdir(pokemon_dir) if f.endswith(('.jpg', '.png'))]
        if images:
            test_image = os.path.join(pokemon_dir, images[0])
            print(f"\nTesting: {pokemon.upper()}")
            print(f"Image: {test_image}")
            print()
            predicted, confidence = predict_pokemon(test_image, top_k=5)
            if predicted:
                match = "✓ CORRECT!" if predicted.lower() == pokemon.lower() else "✗ INCORRECT"
                print(f"Final Prediction: {predicted} ({confidence*100:.2f}%) {match}")
            print("-" * 50)

print("\n✓ Test complete!")
