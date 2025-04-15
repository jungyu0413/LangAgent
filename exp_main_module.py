import cv2
import torch
from torchvision import transforms
from collections import defaultdict

from EXP_module.src.model import NLA_r18
from EXP_module.src.utils import *
from face_det_module.src.util import get_args_parser, get_transform
from face_det_module.src.face_crop import crop


def analyze_expression_changes(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args_parser()
    args.transform = get_transform()
    args.weights_path = '/home/face/Desktop/LangAgent/EXP_module/weights/best.pth'

    model = NLA_r18(args)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.to(device).eval()

    exp_dict = {
        0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happiness',
        4: 'Sadness', 5: 'Anger', 6: 'Neutral'
    }

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.577, 0.4494, 0.4001], std=[0.2628, 0.2395, 0.2383])
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "module": "expression_change",
            "status": "error",
            "expression_change_count": None,
            "analyzed_frames": 0
        }

    prev_label = None
    total_changes = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_tensor, check, box = crop(image_rgb, preprocess, 224, True, device=device)

        if not check:
            current_label = "No Face"
        else:
            with torch.no_grad():
                output = model(output_tensor)
                pred_cls = torch.argmax(output, dim=1).item()
                current_label = exp_dict[pred_cls]

        if (
            prev_label is not None and current_label != prev_label
            and current_label != "No Face" and prev_label != "No Face"
        ):
            total_changes += 1

        prev_label = current_label
        frame_idx += 1

    cap.release()

    return {
        "module": "expression_change",
        "status": "success",
        "expression_change_count": total_changes,
        "analyzed_frames": frame_idx
    }




video_path = '/home/face/Desktop/LangAgent/langchain_demo.mp4'
print(analyze_expression_changes(video_path))