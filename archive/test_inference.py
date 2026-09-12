"""
Simple reproducible test: run repeated inference on a single static image
to check whether predictions are stable or effectively random.

Usage:
  python test_inference.py path/to/image.jpg
"""
import sys
import torch
from PIL import Image
from torchvision import transforms
from model_loader import load_model

IMG_SIZE = 128
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def top1_from_output(output, class_names):
    probs = torch.softmax(output, dim=1)
    top_prob, top_idx = torch.topk(probs, 1, dim=1)
    return class_names[top_idx[0,0].item()], float(top_prob[0,0].item())


def main(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device=device)
    if model is None:
        print("Model not found or failed to load")
        return

    # load class names like camera script does
    import pandas as pd, os
    meta = './pokemon-dataset-1000/metadata.csv'
    if os.path.exists(meta):
        df = pd.read_csv(meta)
        class_names = sorted(df['label'].unique())
    else:
        class_names = [f'Pokemon_{i}' for i in range(1000)]

    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    print(f"Running 20 repeated inferences on {img_path}...")
    results = {}
    with torch.no_grad():
        for i in range(20):
            out = model(tensor)
            name, p = top1_from_output(out, class_names)
            results.setdefault((name, round(p,4)), 0)
            results[(name, round(p,4))] += 1

    print("Observations (top1 -> count):")
    for (name,p), cnt in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name}: {p:.4f}  x{cnt}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python test_inference.py path/to/image.jpg')
    else:
        main(sys.argv[1])
