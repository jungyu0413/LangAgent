import cv2
import json
import torch
import numpy as np
from torchvision import transforms
from torch.nn import Softmax
from collections import defaultdict

from EXP_module.src.model import NLA_r18
from EXP_module.src.utils import *
from face_det_module.src.util import get_args_parser, get_transform
from face_det_module.src.face_crop import crop

def analyze_expression_changes(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
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
        print(f"❌ Could not open video: {video_path}")
        return

    prev_label = None
    total_changes = 0
    change_counter = defaultdict(int)

    print(f"🎬 Processing video: {video_path}")

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

        # Check for label change
        if prev_label is not None and current_label != prev_label:
            total_changes += 1
            change_counter[f"{prev_label} ➜ {current_label}"] += 1

        prev_label = current_label

    cap.release()

    # print(f"\n📊 Total Expression Changes: {total_changes}")
    # print(json.dumps(change_counter, indent=2, ensure_ascii=False))

    return total_changes #, dict(change_counter)


#print(analyze_expression_changes('/home/face/Desktop/LangAgent/langchain_demo.mp4'))