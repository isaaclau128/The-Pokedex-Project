# Pokemon Recognition - Camera Setup Guide

## Overview
This project enables real-time Pokemon recognition using your device's camera. The system uses a CNN model trained in the Jupyter notebook to classify Pokemon in real-time.

## Quick Start

### Step 1: Train the Model (Jupyter Notebook)
First, run the `Pokedex.ipynb` notebook completely. After training finishes, **add this code to the end of your notebook** to save the trained model:

```python
# Save the trained model
torch.save(model.state_dict(), './pokemon_model.pth')
print("✓ Model saved successfully!")
```

This creates a `pokemon_model.pth` file that the camera script will load.

### Step 2: Install Dependencies
Run the setup script to install required packages:

```bash
bash setup_camera.sh
```

Or manually install:
```bash
pip install opencv-python torch torchvision pandas scikit-learn seaborn
```

### Step 3: Run Camera Recognition
```bash
python camera_pokemon_recognizer.py
```

## Controls
- **Q** - Quit the application
- **SPACE** - Pause recognition
- **R** - Resume recognition

## How It Works

### Model Architecture
- **Input**: 128×128 RGB images from camera
- **Architecture**: 
  - Conv1: 3→64 channels + ReLU + MaxPool
  - Conv2: 64→128 channels + ReLU + MaxPool
  - FC1: 8192→256 neurons + Dropout
  - FC2: 256→1000 (Pokemon classes)
- **Output**: Top 3 predictions with confidence scores

### Camera Processing Pipeline
1. **Capture**: Frame from device camera (webcam)
2. **Resize**: Scale to 128×128 pixels
3. **Preprocess**: Convert to RGB and apply normalization
   - Normalization: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
4. **Inference**: Run through trained model
5. **Display**: Show top 3 predictions with confidence percentages
6. **Optimize**: Process every 2nd frame for better real-time performance

## Display Information
The camera window shows:
- **Top 1 Prediction** (Green): Most confident prediction with Pokemon name and confidence %
- **Top 2-3 Predictions** (Orange): Alternative predictions with confidence %
- **Instructions**: Keyboard shortcuts at bottom of frame

## File Structure
```
The-Pokedex-Project/
├── Pokedex.ipynb                    # Main training notebook
├── camera_pokemon_recognizer.py     # Camera recognition script
├── CAMERA_SETUP.md                  # This file
├── setup_camera.sh                  # Dependency installer
├── pokemon_model.pth                # Trained model (generated after training)
└── pokemon-dataset-1000/
    └── metadata.csv                 # Pokemon class labels
```

## Troubleshooting

### "Model not found" Error
**Solution**: The `pokemon_model.pth` file doesn't exist yet. 
1. Run the Jupyter notebook completely
2. Add the model saving code at the end
3. Re-run the camera script

### "Cannot open camera" Error
**Solution**: 
- Check that your device camera is connected
- Check that no other application is using the camera
- On Linux, may need: `sudo apt install libatlas-base-dev`

### Low FPS / Slow Recognition
**Solution**:
- The script processes every 2nd frame by default for performance
- Edit `frame_count % 2` to `frame_count % 5` for faster performance but fewer predictions
- Use GPU if available (script auto-detects)

### Wrong Predictions
**Solution**:
- The model's accuracy depends on training data quality
- Ensure good lighting and clear Pokemon images
- Model works best with images similar to training data

## Advanced Usage

### Custom Confidence Threshold
Edit `camera_pokemon_recognizer.py` to filter predictions by confidence:
```python
min_confidence = 0.3  # Only show predictions > 30% confidence
last_prediction = [p for p in predictions if p['confidence'] > min_confidence]
```

### Adjust Processing Speed
Change the frame skip rate (line ~160):
```python
if frame_count % 2 == 0:  # Process every 2nd frame
    # Change 2 to: 1 (every frame), 5 (skip 4), 10 (skip 9), etc.
```

### Save Recognition Results
Add this code to capture predicted Pokemon:
```python
if key == ord('s'):
    cv2.imwrite(f"pokemon_{frame_count}.jpg", frame)
    if last_prediction:
        print(f"Saved: {last_prediction[0]['name']}")
```

## Dependencies
- **PyTorch**: Deep learning framework
- **OpenCV (cv2)**: Camera and image processing
- **torchvision**: Image transforms
- **pandas**: Data handling
- **numpy**: Numerical operations

## Performance
- GPU (CUDA): ~100 FPS inference
- CPU: ~10-20 FPS inference
- Real-time display: 30 FPS (due to display refresh)

## Notes
- The model processes images at 128×128 resolution
- Class normalization matches ImageNet standards
- Camera input is automatically resized to display at 800×600
- Inference runs on available device (GPU if available, otherwise CPU)

## Support
If you encounter issues:
1. Ensure all dependencies are installed
2. Check that the model file exists and is in the correct directory
3. Verify the metadata.csv file is accessible
4. Check that your camera is working in other applications

Happy Pokemon hunting! 🎮
