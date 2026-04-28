# Complete Setup Guide: Zero to Running Pokemon Camera Recognition

This guide takes you from a fresh start to running real-time Pokemon recognition on your camera.

---

## 📋 Prerequisites
- Python 3.7+ installed
- Device with a camera (webcam)
- ~2GB free disk space
- ~5-10 minutes for first-time setup

---

## 🚀 Complete Step-by-Step Instructions

### Step 1: Navigate to Project Directory
```bash
cd /workspaces/The-Pokedex-Project
```

Verify you're in the right place:
```bash
ls
```

You should see: `Pokedex.ipynb`, `camera_pokemon_recognizer.py`, `model_loader.py`, etc.

---

### Step 2: Install All Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- PyTorch (deep learning framework)
- OpenCV (camera access)
- TorchVision (image processing)
- Pandas (data handling)
- And other required packages

**Time**: ~2-5 minutes (depending on internet speed)

---

### Step 3: Verify Setup (Optional but Recommended)
```bash
python verify_setup.py
```

This checks:
- ✓ Python version
- ✓ All packages installed
- ✓ Camera accessible
- ✓ GPU available (if applicable)
- ✓ Data files present

**Expected output**: Green checkmarks for most items

---

### Step 4A: OPTION A - Use Existing Trained Model

If you already have `pokemon_model.pth` in the directory:

```bash
python -c "from model_loader import model_exists; print('✓ Model ready!' if model_exists() else '✗ No model')"
```

If you see `✓ Model ready!` → **Skip to Step 5**

---

### Step 4B: OPTION B - Train a New Model (First Time Only)

If no model exists yet:

```bash
jupyter notebook Pokedex.ipynb
```

Then in the Jupyter interface:
1. Click **Cell → Run All** (or press Ctrl+A then Shift+Enter)
2. Wait for training to complete (~5-15 minutes depending on GPU)
3. Last cell automatically saves the model as `pokemon_model.pth`
4. Close Jupyter (Ctrl+C in terminal or kernel interrupt)

**Verify model was saved:**
```bash
ls -lh pokemon_model.pth
```

You should see a file ~50-100 MB

---

### Step 5: Verify Model is Loadable
```bash
python model_loader.py
```

**Expected output:**
```
✓ Model loaded from ./pokemon_model.pth
```

If this works, you're ready! ✨

---

### Step 6: Start Camera Recognition
```bash
python camera_pokemon_recognizer.py
```

**Expected behavior:**
- Camera window opens
- Live video feed displays
- Pokemon predictions appear on screen
- Shows top 3 predictions with confidence %

**Controls:**
- **Q** - Quit
- **SPACE** - Pause predictions
- **R** - Resume predictions

---

## 🎯 Quick Command Summary (Copy & Paste)

```bash
# 1. Navigate to project
cd /workspaces/The-Pokedex-Project

# 2. Install everything
pip install -r requirements.txt

# 3. Verify setup (optional)
python verify_setup.py

# 4A. If you have a model, verify it
python model_loader.py

# 4B. If NO model, train it (one time)
jupyter notebook Pokedex.ipynb
# Then: Cell → Run All (in Jupyter interface)

# 5. Run camera recognition
python camera_pokemon_recognizer.py
```

---

## 📊 What Happens at Each Step

| Step | Command | Time | Output |
|------|---------|------|--------|
| 1 | `cd /workspaces/The-Pokedex-Project` | <1s | Directory change |
| 2 | `pip install -r requirements.txt` | 2-5m | Installs all packages |
| 3 | `python verify_setup.py` | 5-10s | Setup verification |
| 4A | `python model_loader.py` | 2-5s | Loads existing model |
| 4B | `jupyter notebook Pokedex.ipynb` | 5-15m | Trains new model |
| 5 | `python camera_pokemon_recognizer.py` | ⚡ Instant | Live camera recognition! |

---

## ✅ Troubleshooting

### "pip: command not found"
```bash
# Try python's pip module instead
python -m pip install -r requirements.txt
```

### "Cannot open camera"
```bash
# Check camera works
python -c "import cv2; cap = cv2.VideoCapture(0); print('✓ Camera OK' if cap.isOpened() else '✗ Camera FAILED')"
```

### "Model not found" when running camera
```bash
# You need to train first
jupyter notebook Pokedex.ipynb
# Then run all cells in Jupyter
```

### "No module named 'torch'"
```bash
# Reinstall PyTorch specifically
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### "Slow performance"
- **On CPU**: Normal (8-15 FPS). To speed up, skip more frames: Edit `camera_pokemon_recognizer.py` line ~160, change `% 2` to `% 1`
- **Has GPU**: Should be ~100 FPS. Check with: `nvidia-smi`

---

## 🔄 Repeat Usage (After First Time)

Once set up, you only need:

```bash
cd /workspaces/The-Pokedex-Project
python camera_pokemon_recognizer.py
```

That's it! Model loads instantly from disk.

---

## 📁 File Locations

| File | Purpose |
|------|---------|
| `requirements.txt` | Package list |
| `Pokedex.ipynb` | Training notebook |
| `camera_pokemon_recognizer.py` | Camera app |
| `model_loader.py` | Model loading helper |
| `pokemon_model.pth` | Trained model (created after training) |
| `pokemon-dataset-1000/` | Training data |

---

## 🎓 Understanding the Workflow

```
FIRST TIME:
├─ Install packages → 2-5 min
├─ Train model → 5-15 min
└─ Run camera → Instant start

SUBSEQUENT TIMES:
├─ Run camera → Instant start (model loads from disk)
└─ Enjoy! 🎮
```

---

## 💡 Tips

1. **First run takes time**: Training takes 5-15 minutes. Subsequent runs with the same model are instant.

2. **GPU is faster**: If you have NVIDIA GPU, PyTorch will auto-detect and use it (100x faster inference).

3. **Model persists**: Once `pokemon_model.pth` is created, it stays on disk. You never need to retrain unless you want a better model.

4. **Can move files**: You can copy `pokemon_model.pth` to another machine. Just run:
   ```bash
   pip install -r requirements.txt
   python camera_pokemon_recognizer.py
   ```

5. **Pause to debug**: Hit SPACE to pause at any time. Hit R to resume.

---

## 🆘 Getting Help

If stuck, check:
- See `CAMERA_SETUP.md` for detailed troubleshooting
- Run `python verify_setup.py` to diagnose issues
- Check `LOAD_MODEL_GUIDE.md` for model-specific questions

---

## ✨ Final Checklist

- [ ] Python 3.7+ installed
- [ ] In project directory: `cd /workspaces/The-Pokedex-Project`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Setup verified: `python verify_setup.py` (all green)
- [ ] Model exists/trained: `python model_loader.py` (shows ✓)
- [ ] Camera works: `python camera_pokemon_recognizer.py` (window opens)
- [ ] Ready to recognize Pokemon! 🎮

---

## 🎬 Quick Video Flow

```
Terminal 1: Install
$ pip install -r requirements.txt
✓ Done

Terminal 2: Train (one time)
$ jupyter notebook Pokedex.ipynb
[Cell → Run All in browser]
✓ Model saved

Terminal 3: Run Camera  
$ python camera_pokemon_recognizer.py
✓ Camera opens, shows predictions
```

---

**You're all set!** Follow the steps above and you'll have Pokemon camera recognition running in ~10-20 minutes. 🚀
