import cv2
import numpy as np
import torch
from torchvision import transforms

from VA_module.src.resnet import VA_Model
from face_det_module.src.util import get_args_parser, get_transform, pre_trained_wegiths_load
from face_det_module.src.face_crop import crop


def compute_va_change_average(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args_parser()
    args.transform = get_transform()
    args.weights_path = '/home/face/Desktop/LangAgent/VA_module/weights/best.pth'

    model = VA_Model(args)
    cp = torch.load(args.weights_path, map_location=device)
    model = pre_trained_wegiths_load(model, cp)
    model = model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.577, 0.4494, 0.4001],
                             std=[0.2628, 0.2395, 0.2383])
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "module": "va_change",
            "status": "error",
            "valence_change": None,
            "arousal_change": None,
            "valid_frames": 0,
            "total_frames": 0
        }

    prev_val, prev_aro = None, None
    val_diffs, aro_diffs = [], []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_tensor, check, box = crop(frame_rgb, preprocess, 224, True, device=device)

        if check:
            with torch.no_grad():
                _, pred_val, pred_aro, _, _ = model(output_tensor)
                val = np.clip(pred_val.item(), -1, 1)
                aro = np.clip(pred_aro.item(), -1, 1)

            if prev_val is not None:
                val_diffs.append(abs(val - prev_val))
                aro_diffs.append(abs(aro - prev_aro))

            prev_val, prev_aro = val, aro

        frame_idx += 1

    cap.release()

    if len(val_diffs) == 0:
        return {
            "module": "va_change",
            "status": "no_face_detected",
            "valence_change": None,
            "arousal_change": None,
            "valid_frames": 0,
            "total_frames": frame_idx
        }

    return {
        "module": "va_change",
        "status": "success",
        "valence_change": float(np.mean(val_diffs)),
        "arousal_change": float(np.mean(aro_diffs)),
        "valid_frames": len(val_diffs),
        "total_frames": frame_idx
    }


# video_path = '/home/face/Desktop/LangAgent/langchain_demo.mp4'
# print(compute_va_change_average(video_path))