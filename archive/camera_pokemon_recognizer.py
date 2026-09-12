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
import platform
import json

from torchvision import models

try:
    import AVFoundation  # type: ignore[import-not-found]
    AVFOUNDATION_AVAILABLE = True
except Exception:
    AVFOUNDATION_AVAILABLE = False

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


def create_resnet50_model(num_classes=1000):
    """Create a ResNet50 classifier without downloading pretrained weights."""
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ===================== Configuration =====================
IMG_SIZE = 224  # ResNet50 standard input size (was 128; larger = better accuracy for ResNet50)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./pokemon_model.pth"
METADATA_PATH = "./pokemon-dataset-1000/metadata.csv"
TEMPERATURE_PATH = "./temperature.json"

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
        # Prefer an explicit classes.json saved at training time to ensure
        # the same ordering/mapping used during training is restored.
        classes_json = './classes.json'
        if os.path.exists(classes_json):
            with open(classes_json, 'r') as f:
                class_names = json.load(f)
                print(f"✓ Loaded class names from {classes_json}")
                return class_names

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


def load_model(model_path, device, num_classes=1000):
    """Load the trained model"""
    try:
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found at {model_path}")
            print("Please run the Jupyter notebook to train and save the model first.")
            print("Add this code to the notebook to save the model:")
            print("  torch.save(model.state_dict(), './pokemon_model.pth')")
            return None
        
        state_dict = torch.load(model_path, map_location=device)

        if any(key.startswith("layer1.") or key.startswith("bn1.") for key in state_dict.keys()):
            model = create_resnet50_model(num_classes=num_classes).to(device)
            architecture = "ResNet50"
        else:
            model = CNNClassifier(img_size=IMG_SIZE, num_classes=num_classes).to(device)
            architecture = "CNN"

        model.load_state_dict(state_dict)
        model.eval()
        print(f"✓ Model loaded successfully from {model_path} ({architecture})")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_temperature(temperature_path):
    """Load a calibrated temperature from disk, if available."""
    try:
        if os.path.exists(temperature_path):
            with open(temperature_path, 'r') as f:
                payload = json.load(f)
                temperature = float(payload.get('temperature', 1.0))
                if temperature > 0:
                    print(f"✓ Loaded calibrated temperature: {temperature:.4f}")
                    return temperature
    except Exception as e:
        print(f"Warning: could not load temperature calibration: {e}")

    return 1.0


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


def get_top_predictions(output, class_names, top_k=3, temperature=1.0):
    """Get top K predictions with probabilities"""
    scaled_output = output / temperature
    probabilities = torch.softmax(scaled_output, dim=1)
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

    predictions = []
    for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
        name = class_names[idx] if idx < len(class_names) else f"Pokemon_{idx}"
        predictions.append({
            'name': name,
            'confidence': prob,
            'index': int(idx)
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


def get_preferred_camera_indices():
    """Return camera indices ordered with the built-in MacBook camera first."""
    indices = [0, 1, 2, 3]
    if not AVFOUNDATION_AVAILABLE or platform.system() != "Darwin":
        return indices

    try:
        devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
        ranked_indices = []
        fallback_indices = []

        for index, device in enumerate(devices):
            name = str(device.localizedName())
            lower_name = name.lower()

            if any(token in lower_name for token in ["macbook", "built-in", "facetime hd"]):
                ranked_indices.append(index)
            elif any(token in lower_name for token in ["iphone", "continuity", "external"]):
                fallback_indices.append(index)
            else:
                fallback_indices.append(index)

        ordered = ranked_indices + [index for index in fallback_indices if index not in ranked_indices]
        if ordered:
            return ordered
    except Exception as e:
        print(f"Camera device discovery failed, falling back to default order: {e}")

    return indices


def initialize_camera():
    """Open the first working camera and warm it up."""
    camera_indices = get_preferred_camera_indices()
    if platform.system() == "Darwin":
        backends = [cv2.CAP_AVFOUNDATION, None]
    else:
        backends = [None]

    for camera_index in camera_indices:
        for backend in backends:
            cap = cv2.VideoCapture(camera_index) if backend is None else cv2.VideoCapture(camera_index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            # Give the camera a moment to produce its first valid frame.
            for _ in range(5):
                cap.read()

            ret, _ = cap.read()
            if ret:
                if camera_index != 0:
                    print(f"✓ Camera opened successfully on index {camera_index}")
                return cap

            cap.release()

    return None


# ===================== Main Camera Loop =====================
def main():
    print("=" * 50)
    print("Pokemon Recognition - Camera Mode")
    print("=" * 50)
    
    # Load class names
    class_names = load_class_names(METADATA_PATH)
    print(f"✓ Loaded {len(class_names)} Pokemon classes")

    # Load model after class names so the classifier head matches the dataset order.
    model = load_model(MODEL_PATH, DEVICE, num_classes=len(class_names))
    if model is None:
        print("Cannot start camera mode without a trained model.")
        sys.exit(1)

    # Load calibrated temperature if present.
    temperature = load_temperature(TEMPERATURE_PATH)
    
    # Initialize camera
    cap = initialize_camera()
    if cap is None:
        print("Error: Cannot open camera. Make sure a camera is connected.")
        sys.exit(1)
    
    print("✓ Camera opened successfully")
    print("\nStarting real-time Pokemon recognition...")
    print("Controls: Q = Quit | SPACE = Pause | R = Resume\n")
    
    paused = False
    last_prediction = None
    frame_count = 0
    frozen_frame = None
    
    try:
        while True:
            if paused and frozen_frame is not None:
                frame = frozen_frame.copy()
            else:
                ret, frame = cap.read()
                if not ret:
                    # Retry a few times before giving up; some macOS cameras need brief recovery time.
                    retry_success = False
                    for _ in range(5):
                        ret, frame = cap.read()
                        if ret:
                            retry_success = True
                            break

                    if not retry_success:
                        print("Error: Cannot read from camera")
                        break

                if not paused:
                    frozen_frame = frame.copy()
            
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
                            last_prediction = get_top_predictions(
                                output,
                                class_names,
                                top_k=3,
                                temperature=temperature,
                            )
                    
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

                # Keep the most recent predictions visible while paused.
                if last_prediction:
                    display_frame = draw_predictions(display_frame, last_prediction)
            
            # Display frame
            cv2.imshow("Pokemon Recognition", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting...")
                break
            elif key == ord(' '):
                paused = True
                if frozen_frame is None:
                    frozen_frame = frame.copy()
            elif key == ord('r'):
                paused = False
                frozen_frame = None
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed. Goodbye!")


if __name__ == "__main__":
    main()
