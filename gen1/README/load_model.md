# Load the gen1 Model

Use the local gen1 loader in scripts:

```python
from model_loader import load_model, model_exists

if model_exists():
    model, class_names, device = load_model()
```

The loader expects:

- `gen1/pokemon_model_gen1_resnet50.pth`
- `gen1/gen1_classes.json`

It rebuilds the same ResNet50 classifier used in `gen1/gen1.ipynb` and loads the saved weights without retraining.