# Camera Setup for gen1

## Run Order

1. Train or load the Gen 1 model in `gen1/gen1.ipynb`.
2. Confirm these files exist in `gen1/`:
   - `pokemon_model_gen1_resnet50.pth`
   - `gen1_classes.json`
3. Install dependencies:

```bash
bash gen1/setup_camera.sh
```

4. Start the camera recognizer:

```bash
python gen1/camera_pokemon_recognizer.py
```

## Controls

- `Q`: Quit
- `SPACE`: Pause or resume predictions

## Troubleshooting

### Camera does not open

- Check macOS camera permissions for Terminal and VS Code.
- Close FaceTime, Photo Booth, Zoom, or any other app using the camera.

### Model not found

- Run the save cell in `gen1/gen1.ipynb`.
- Make sure the model file is `gen1/pokemon_model_gen1_resnet50.pth`.

### SSL or pretrained weight download issues

- The notebook and loader already use `certifi` for the ResNet50 checkpoint download.
- If needed, install or refresh it with `python -m pip install certifi`.