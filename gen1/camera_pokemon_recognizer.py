from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
from torchvision import transforms

from model_loader import IMG_SIZE, load_model, model_exists


FRAME_SKIP = 2
TOP_K = 3
WINDOW_NAME = 'Gen 1 Pokedex Camera'
MAX_CONSECUTIVE_READ_FAILURES = 30
READ_RETRY_COUNT = 5
NULL_LABEL = 'NULL'
NULL_CONFIDENCE_THRESHOLD = 0.65
NULL_MARGIN_THRESHOLD = 0.10
SMOOTHING_WINDOW = 5


def build_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def preprocess_frame(frame_bgr, transform, device):
    height, width = frame_bgr.shape[:2]
    crop_size = min(height, width)
    top = max((height - crop_size) // 2, 0)
    left = max((width - crop_size) // 2, 0)
    cropped_frame = frame_bgr[top:top + crop_size, left:left + crop_size]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    tensor = transform(cropped_rgb).unsqueeze(0).to(device)
    return tensor


def predict_top_k(model, input_tensor, class_names, k=TOP_K, probability_history=None):
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

    if probability_history is not None:
        probability_history.append(probabilities.squeeze(0).cpu())
        stacked_probabilities = torch.stack(list(probability_history), dim=0)
        probabilities = stacked_probabilities.mean(dim=0, keepdim=True)

    confidences, indices = torch.topk(probabilities, k=min(k, len(class_names)), dim=1)

    results = []
    for confidence, index in zip(confidences[0].cpu().tolist(), indices[0].cpu().tolist()):
        results.append((class_names[index], confidence))
    return results


def apply_null_rule(predictions):
    if not predictions:
        return [(NULL_LABEL, 1.0)]

    top_name, top_confidence = predictions[0]
    second_confidence = predictions[1][1] if len(predictions) > 1 else 0.0
    confidence_margin = top_confidence - second_confidence

    if top_confidence < NULL_CONFIDENCE_THRESHOLD or confidence_margin < NULL_MARGIN_THRESHOLD:
        return [(NULL_LABEL, 1.0 - top_confidence)] + predictions

    return predictions


def draw_predictions(frame, predictions, paused):
    title = 'PAUSED' if paused else 'LIVE'
    cv2.putText(frame, f'{title}  |  Q quit  SPACE pause/resume', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, 'Center the Pokemon in frame for best results', (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for row, (name, confidence) in enumerate(predictions):
        color = (0, 0, 255) if name == NULL_LABEL else ((0, 255, 0) if row == 0 else (0, 165, 255))
        label = f'{row + 1}. {name}: {confidence * 100:.1f}%'
        cv2.putText(frame, label, (20, 95 + (row * 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def open_camera(camera_index=0):
    capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError('Could not open camera. Check macOS camera permissions and whether another app is using it.')
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def read_frame(capture, retries=READ_RETRY_COUNT):
    for _ in range(retries):
        ok, frame = capture.read()
        if ok and frame is not None and getattr(frame, 'size', 0) > 0:
            return True, frame
    return False, None


def main():
    if not model_exists():
        model_path = Path(__file__).resolve().parent / 'pokemon_model_gen1_resnet50.pth'
        raise FileNotFoundError(f'Model not found at {model_path}')

    model, class_names, device = load_model()
    transform = build_transform()
    capture = open_camera()

    paused = False
    frame_count = 0
    last_predictions = []
    consecutive_failures = 0
    probability_history = deque(maxlen=SMOOTHING_WINDOW)

    print('Camera started.')
    print('Controls: Q = quit, SPACE = pause/resume')

    try:
        while True:
            ok, frame = read_frame(capture)
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    raise RuntimeError('Failed to read frames from camera repeatedly. Check macOS camera permissions and whether another app is interrupting the feed.')
                continue

            consecutive_failures = 0

            frame_count += 1
            display_frame = frame.copy()

            if not paused and frame_count % FRAME_SKIP == 0:
                input_tensor = preprocess_frame(frame, transform, device)
                raw_predictions = predict_top_k(
                    model,
                    input_tensor,
                    class_names,
                    probability_history=probability_history,
                )
                last_predictions = apply_null_rule(raw_predictions)

            draw_predictions(display_frame, last_predictions, paused)
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == 32:
                paused = not paused

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()