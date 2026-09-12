"""
Camera Frame Collection & Labeling Tool
Capture frames from camera and label them interactively for domain adaptation training.

Usage:
    python collect_camera_frames.py
    
Controls:
    SPACE  = Capture and label frame
    Q      = Quit
    ESC    = Skip current frame without saving
"""

import cv2
import os
import json
from datetime import datetime
from pathlib import Path

# ===================== Configuration =====================
OUTPUT_DIR = "./camera_frames_for_finetuning"
CLASSES_JSON = "./classes.json"

# Create output directory structure
os.makedirs(OUTPUT_DIR, exist_ok=True)
metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")

# Load class names for autocomplete
class_names = []
if os.path.exists(CLASSES_JSON):
    with open(CLASSES_JSON, 'r') as f:
        class_names = json.load(f)
    print(f"✓ Loaded {len(class_names)} Pokemon classes")
else:
    print(f"⚠ {CLASSES_JSON} not found. You'll type class names manually.")

# Initialize or load metadata
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Resuming from {len(metadata)} existing labeled frames")
else:
    metadata = []

# ===================== Camera Setup =====================
def initialize_camera():
    """Open camera with fallback logic."""
    for camera_idx in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            # Warm up camera
            for _ in range(5):
                cap.read()
            if camera_idx != 0:
                print(f"✓ Camera opened on index {camera_idx}")
            return cap
    return None

# ===================== Main Loop =====================
def main():
    cap = initialize_camera()
    if cap is None:
        print("❌ Error: Cannot open camera. Make sure a camera is connected.")
        return
    
    print("\n" + "="*70)
    print("Camera Frame Collection Tool")
    print("="*70)
    print("Controls:")
    print("  SPACE = Capture & label frame")
    print("  ESC   = Skip frame")
    print("  Q     = Quit")
    print("="*70 + "\n")
    
    frame_count = len(metadata)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read from camera")
                break
            
            # Display camera frame
            display_frame = cv2.resize(frame, (800, 600))
            
            # Add header text
            cv2.putText(
                display_frame,
                f"Frames Labeled: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.putText(
                display_frame,
                "SPACE=Capture | ESC=Skip | Q=Quit",
                (10, 600 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )
            
            cv2.imshow("Camera Frame Collector", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == 27:  # ESC
                print("Skipped frame")
                continue
            elif key == ord(' '):  # SPACE
                # Prompt for label
                label = input_label(class_names)
                if label:
                    # Save frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    frame_path = os.path.join(OUTPUT_DIR, f"{label}_{timestamp}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    # Record metadata
                    metadata.append({
                        "image_path": f"{label}_{timestamp}.jpg",
                        "label": label,
                        "timestamp": timestamp
                    })
                    
                    # Save metadata
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    frame_count += 1
                    print(f"✓ Saved frame as '{label}_{timestamp}.jpg' (Total: {frame_count})")
                else:
                    print("Label skipped")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Collection complete: {frame_count} frames labeled")
        print(f"✓ Frames saved in: {OUTPUT_DIR}")
        print(f"✓ Run: python finetune_on_camera_frames.py")


def input_label(class_names):
    """Get label from user with basic autocomplete."""
    print("\nEnter Pokemon name (or 'skip' to skip):")
    if class_names:
        print(f"Available: {', '.join(class_names[:20])}{'...' if len(class_names) > 20 else ''}")
    
    label = input("→ ").strip()
    
    if label.lower() == 'skip':
        return None
    
    if class_names and label not in class_names:
        print(f"⚠ '{label}' not in known classes. Proceeding anyway.")
    
    return label


if __name__ == "__main__":
    main()
