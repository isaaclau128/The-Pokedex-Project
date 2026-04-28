"""
Real-time Pokemon Recognition using Device Camera
Uses the trained CNN model from the Jupyter notebook
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import sys

# ===================== Model Definition =====================
class CNNClassifier(nn.Module):
    def __init__(self, img_size=128, num_classes=1000):
        super(CNNClassifier, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * (img_size // 4) * (img_size // 4), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ===================== Configuration =====================
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./pokemon_model.pth"
METADATA_PATH = "./pokemon-dataset-1000/metadata.csv"

# Image preprocessing transforms (must match training transforms)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


# ===================== Helper Functions =====================
def load_class_names(metadata_path):
    """Load Pokemon class names from metadata CSV"""
    try:
        if os.path.exists(metadata_path):
            df = pd.read_csv(metadata_path)
            # Get unique class names sorted (assumes 'label' column exists)
            if 'label' in df.columns:
                class_names = sorted(df['label'].unique())
                return class_names
    except Exception as e:
        print(f"Error loading metadata: {e}")
    
    # Fallback: create generic class names
    return [f"Pokemon_{i}" for i in range(1000)]


def load_model(model_path, device):
    """Load the trained model"""
    try:
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found at {model_path}")
            print("Please run the Jupyter notebook to train and save the model first.")
            print("Add this code to the notebook to save the model:")
            print("  torch.save(model.state_dict(), './pokemon_model.pth')")
            return None
        
        model = CNNClassifier(img_size=IMG_SIZE, num_classes=1000).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✓ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def preprocess_frame(frame, transform, device):
    """Convert camera frame to model input"""
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame.astype('uint8'), 'RGB')
        
        # Apply transforms
        tensor_image = transform(pil_image)
        
        # Add batch dimension
        tensor_image = tensor_image.unsqueeze(0).to(device)
        
        return tensor_image
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return None


def get_top_predictions(output, class_names, top_k=3):
    """Get top K predictions with probabilities"""
    probabilities = torch.softmax(output, dim=1)
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    predictions = []
    for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
        predictions.append({
            'name': class_names[idx],
            'confidence': prob,
            'index': idx
        })
    
    return predictions


def draw_predictions(frame, predictions):
    """Draw predictions on the frame"""
    h, w = frame.shape[:2]
    
    # Draw semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw predictions
    y_offset = 40
    for i, pred in enumerate(predictions):
        text = f"{i+1}. {pred['name']}: {pred['confidence']:.1%}"
        color = (0, 255, 0) if i == 0 else (0, 165, 255)  # Green for top, orange for others
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2)
        y_offset += 30
    
    # Add instructions
    cv2.putText(frame, "Press Q to quit | SPACE to pause", (10, h-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame


# ===================== Main Camera Loop =====================
def main():
    print("=" * 50)
    print("Pokemon Recognition - Camera Mode")
    print("=" * 50)
    
    # Load model
    model = load_model(MODEL_PATH, DEVICE)
    if model is None:
        print("Cannot start camera mode without a trained model.")
        sys.exit(1)
    
    # Load class names
    class_names = load_class_names(METADATA_PATH)
    print(f"✓ Loaded {len(class_names)} Pokemon classes")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera. Make sure a camera is connected.")
        sys.exit(1)
    
    print("✓ Camera opened successfully")
    print("\nStarting real-time Pokemon recognition...")
    print("Controls: Q = Quit | SPACE = Pause | R = Resume\n")
    
    paused = False
    last_prediction = None
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read from camera")
                break
            
            # Resize for display (optional, for faster processing)
            display_frame = cv2.resize(frame, (800, 600))
            
            if not paused:
                # Run inference every few frames for better performance
                frame_count += 1
                if frame_count % 2 == 0:  # Process every 2nd frame
                    try:
                        # Preprocess frame
                        input_tensor = preprocess_frame(frame, transform, DEVICE)
                        
                        if input_tensor is not None:
                            # Run inference
                            with torch.no_grad():
                                output = model(input_tensor)
                            
                            # Get predictions
                            last_prediction = get_top_predictions(output, class_names, top_k=3)
                    
                    except Exception as e:
                        print(f"Inference error: {e}")
                
                # Draw last predictions
                if last_prediction:
                    display_frame = draw_predictions(display_frame, last_prediction)
            
            else:
                # Show paused message
                cv2.putText(display_frame, "PAUSED - Press R to Resume", 
                           (display_frame.shape[1]//2 - 150, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow("Pokemon Recognition", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting...")
                break
            elif key == ord(' '):
                paused = True
            elif key == ord('r'):
                paused = False
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed. Goodbye!")


if __name__ == "__main__":
    main()
