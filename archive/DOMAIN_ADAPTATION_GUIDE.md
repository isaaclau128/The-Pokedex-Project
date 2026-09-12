# Domain Gap Fix: Camera Fine-tuning Workflow

## Problem
The model achieves 0.7 accuracy on the validation dataset, but performs poorly on live camera input. This is a **domain gap** issue — training images differ significantly from camera feeds (lighting, angles, backgrounds, blur).

## Solution
Fine-tune the pre-trained model on **real camera frames** to adapt it to your specific camera and environment.

---

## Quick Start (3 steps)

### Step 1: Collect Labeled Camera Frames
Run the frame collection tool and capture ~100-200 labeled Pokemon images from your camera:

```bash
python collect_camera_frames.py
```

**What it does:**
- Opens your camera and displays live feed
- You press SPACE to capture a frame, then type the Pokemon name
- Frames are saved in `camera_frames_for_finetuning/` with labels

**Controls:**
- `SPACE` = Capture frame & label it
- `ESC` = Skip frame  
- `Q` = Quit

**Tips:**
- Capture Pokemon from different angles, distances, lighting
- Include variations: daylight, indoor lighting, various backgrounds
- Aim for 100-200 frames (10-20 per Pokemon species for best results)
- Type lowercase Pokemon names exactly as they appear in the dataset

### Step 2: Fine-tune Model on Camera Frames
After collecting frames, run the fine-tuning script:

```bash
python finetune_on_camera_frames.py
```

**What it does:**
- Loads your pre-trained `pokemon_model.pth`
- Trains on collected camera frames with a low learning rate (1e-4)
- Saves improved model back to `pokemon_model.pth`
- Creates backup of original as `pokemon_model_pretrain_[timestamp].pth`

**Expected:**
- Fine-tuning takes ~2-5 minutes
- Validation accuracy should improve to >85% on camera frames
- Training loss decreases smoothly

### Step 3: Test Camera Detection
Test the fine-tuned model:

```bash
python camera_pokemon_recognizer.py
```

**Expected result:** Detection should be much better on live camera.

---

## How It Works

**Before fine-tuning:**
```
Dataset images (fixed angle, clean background, controlled lighting)
         ↓
    ResNet50 model
         ↓
Camera input (variable angle, messy background, real lighting)
         ↓
    POOR DETECTION ❌
```

**After fine-tuning:**
```
Dataset images + Real camera frames (same model!)
         ↓
    ResNet50 model (adapted to camera)
         ↓
Camera input (same camera used for fine-tuning)
         ↓
    GOOD DETECTION ✓
```

---

## Troubleshooting

### "Metadata not found" when running `finetune_on_camera_frames.py`
- Run `collect_camera_frames.py` first and capture at least 20 frames

### Still poor detection after fine-tuning
- Collect more frames (aim for 200+)
- Ensure frame variety (different angles, lighting, distances)
- Check that Pokemon names match the class names in `classes.json`

### Want to collect more frames later?
- Run `collect_camera_frames.py` again — it will resume from existing frames
- Run `finetune_on_camera_frames.py` again to incorporate new frames

### Want to reset and start over?
```bash
rm -rf camera_frames_for_finetuning/
# Then run collect_camera_frames.py again
```

---

## Technical Details

- **Learning rate:** 1e-4 (very low to avoid "catastrophic forgetting")
- **Batch size:** 16 (small batch for stable fine-tuning)
- **Epochs:** 5 (typically enough with low LR)
- **Data augmentation:** Light (5° rotation, ±0.1 color jitter) — preserves camera domain
- **Train/val split:** 80/20

The low learning rate ensures the model retains knowledge from the large dataset while adapting specifically to your camera's characteristics.

---

## Example Timeline

```
1. Run collect_camera_frames.py
   → Capture 150 frames over ~10 minutes
   
2. Run finetune_on_camera_frames.py  
   → Fine-tunes in ~3 minutes
   → Shows progress: Epoch 1/5, 2/5, ...
   
3. Run camera_pokemon_recognizer.py
   → Test with live camera
   → Detection should be much better
```

---

## Next Steps If Still Not Working

If detection is still poor after fine-tuning:

1. **Collect more frames** (200-300) with more variety
2. **Check frame quality** — ensure frames are clear and well-lit
3. **Verify class names** — make sure you're typing Pokemon names correctly
4. **Retrain with IMG_SIZE=256** for even better quality (edit scripts to increase IMG_SIZE)

Or reach out with details about which Pokemon are still misdetected.
