# Quick Reference - Pokemon Camera Recognition

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model & Save
Open `Pokedex.ipynb`, run all cells, then add at end:
```python
torch.save(model.state_dict(), './pokemon_model.pth')
```

### 3. Run Camera
```bash
python camera_pokemon_recognizer.py
```

---

## ⌨️ Camera Controls
| Key | Action |
|-----|--------|
| **Q** | Quit application |
| **SPACE** | Pause recognition |
| **R** | Resume recognition |

---

## 🔧 Verification & Troubleshooting

### Check Setup
```bash
python verify_setup.py
```

### Install Camera Dependencies Only
```bash
bash setup_camera.sh
```

### Check Camera Access
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('✓ Camera OK' if cap.isOpened() else '✗ Camera FAILED')"
```

---

## 📊 Files Reference
| File | Purpose |
|------|---------|
| `Pokedex.ipynb` | Model training notebook |
| `camera_pokemon_recognizer.py` | Live camera recognition |
| `verify_setup.py` | Setup verification tool |
| `pokemon_model.pth` | Trained model (after training) |
| `CAMERA_SETUP.md` | Detailed documentation |
| `GETTING_STARTED.md` | Complete setup guide |
| `MODEL_SAVE_TEMPLATE.py` | Code to add to notebook |

---

## 🎛️ Performance Tuning

### Speed Up Recognition (Process Every Frame)
In `camera_pokemon_recognizer.py`, line ~160:
```python
if frame_count % 1 == 0:  # Changed from % 2
```

### Slow Down for CPU (Skip More Frames)
```python
if frame_count % 5 == 0:  # Process every 5th frame
```

---

## 🔍 Common Issues

| Issue | Solution |
|-------|----------|
| "Model not found" | Train notebook & save model with torch.save() |
| "Cannot open camera" | Check camera connection, close other apps using it |
| Slow recognition | Normal on CPU, try frame skipping |
| Wrong predictions | Ensure good lighting, clear Pokemon images |
| ImportError | Run `pip install -r requirements.txt` |

---

## 📈 Model Info
- **Input**: 128×128 RGB images
- **Classes**: 1000 Pokemon species  
- **Speed**: GPU ~100 FPS, CPU ~15 FPS
- **Accuracy**: Depends on training (check notebook)

---

## 🎯 Next: Advanced Usage
See `CAMERA_SETUP.md` for:
- Custom confidence filtering
- Recording predictions
- Saving snapshot images
- Performance optimization
- GPU configuration

---

**Last Updated**: 2024
**Version**: 1.0
