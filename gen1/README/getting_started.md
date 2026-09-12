# Getting Started with the gen1 Camera Recognizer

## What You Need

- `gen1/PokemonData/`
- `gen1/pokemon_model_gen1_resnet50.pth`
- `gen1/gen1_classes.json`

## Setup

```bash
bash gen1/setup_camera.sh
```

## Start the Camera

```bash
python gen1/camera_pokemon_recognizer.py
```

## Controls

- `Q`: Quit
- `SPACE`: Pause or resume predictions

## Notes

- This version is wired to the Gen 1 ResNet50 notebook in `gen1/gen1.ipynb`.
- It does not move or depend on the root-level camera files.