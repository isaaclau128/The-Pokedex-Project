#!/usr/bin/env bash

set -euo pipefail

python -m pip install --upgrade pip
python -m pip install torch torchvision opencv-python pillow numpy certifi

echo 'Camera dependencies installed for gen1.'
echo 'Run: python gen1/camera_pokemon_recognizer.py'