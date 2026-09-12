#!/usr/bin/env python3
"""
Setup Verification Script
Checks that all dependencies are installed and camera is accessible
Run this before training the model to ensure everything is set up correctly
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    print("Checking Python version...", end=" ")
    if sys.version_info >= (3, 7):
        print("✓")
        print(f"  Python {sys.version.split()[0]}")
        return True
    else:
        print("✗ FAILED")
        print(f"  Python {sys.version_info.major}.{sys.version_info.minor} detected. Need 3.7+")
        return False

def check_imports():
    """Check that all required packages can be imported"""
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'seaborn': 'Seaborn',
        'PIL': 'Pillow'
    }
    
    print("\nChecking packages...")
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_camera():
    """Check that camera is accessible"""
    print("\nChecking camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("  ✓ Camera working")
                return True
            else:
                print("  ✗ Camera opened but cannot read frames")
                return False
        else:
            print("  ✗ Camera not found or not accessible")
            print("    Ensure camera is connected and no other app is using it")
            return False
    except Exception as e:
        print(f"  ✗ Camera check failed: {e}")
        return False

def check_gpu():
    """Check for GPU availability"""
    print("\nChecking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ GPU found: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("  ℹ GPU not available (CPU will be used)")
            return False
    except Exception as e:
        print(f"  ℹ GPU check inconclusive: {e}")
        return True  # Not a failure, just info

def check_data_files():
    """Check if data files exist"""
    print("\nChecking data files...")
    
    files = {
        './pokemon-dataset-1000/metadata.csv': 'Pokemon metadata',
        './Pokedex.ipynb': 'Training notebook'
    }
    
    all_ok = True
    for path, name in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024
            print(f"  ✓ {name} ({size:.1f} MB)")
        else:
            print(f"  ✗ {name} - NOT FOUND at {path}")
            all_ok = False
    
    return all_ok

def check_model_file():
    """Check if trained model exists"""
    print("\nChecking trained model...")
    model_path = './pokemon_model.pth'
    
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / 1024 / 1024
        print(f"  ✓ Model found ({size:.1f} MB)")
        return True
    else:
        print(f"  ℹ Model not found - You'll need to train it first")
        print(f"    1. Run: jupyter notebook Pokedex.ipynb")
        print(f"    2. Run all cells to train the model")
        print(f"    3. Add model saving code at the end")
        print(f"    4. Then run: python camera_pokemon_recognizer.py")
        return False

def main():
    print("=" * 50)
    print("Pokemon Camera Recognition - Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Required Packages", check_imports()),
        ("Camera Hardware", check_camera()),
        ("GPU Support", check_gpu()),
        ("Data Files", check_data_files()),
        ("Trained Model", check_model_file())
    ]
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in checks[:-1] if result)  # Exclude model check
    total = len(checks) - 1
    
    for name, result in checks[:-1]:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nModel Status: ", end="")
    if checks[-1][1]:
        print("✓ Ready (can run camera script)")
    else:
        print("⚠ Not trained yet (see instructions above)")
    
    print("\n" + "=" * 50)
    
    if passed == total:
        print("✓ Setup is complete and ready!")
        print("\nNext steps:")
        print("1. Train the model: jupyter notebook Pokedex.ipynb")
        print("2. Save the model (add code at end of notebook)")
        print("3. Run camera: python camera_pokemon_recognizer.py")
        return 0
    else:
        print(f"✗ {total - passed} check(s) failed")
        print("\nPlease fix the issues above and run this script again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
