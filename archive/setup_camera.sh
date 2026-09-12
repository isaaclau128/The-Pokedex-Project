#!/bin/bash

# Install required packages for camera-based Pokemon recognition

echo "Installing camera recognition dependencies..."

pip install opencv-python --quiet
pip install torch torchvision --quiet
echo "✓ Core packages installed"

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Run the Jupyter notebook (Pokedex.ipynb) to train the model"
echo "2. After training completes, add this code to the notebook to save the model:"
echo "   torch.save(model.state_dict(), './pokemon_model.pth')"
echo "3. Then run the camera script:"
echo "   python camera_pokemon_recognizer.py"
