import cv2
import numpy as np
import torch
from torchvision import transforms
from VA_module.src.resnet import VA_Model
from face_det_module.src.util import get_args_parser, get_transform, pre_trained_wegiths_load
from face_det_module.src.face_crop import crop

def compute_va_change_average(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    args = get_args_parser()
    args.transform = get_transform()
    args.weights_path = '/home/face/Desktop/LangAgent/VA_module/weights/best.pth'

    model = VA_Model(args)
    cp = torch.load(args.weights_path, map_location=device)
    model = pre_trained_wegiths_load(model, cp)
    model = model.to(device).eval()

    # 전처리
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.577, 0.4494, 0.4001],
                             std=[0.2628, 0.2395, 0.2383])
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    prev_val, prev_aro = None, None
    val_diffs, aro_diffs = [], []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_tensor, check, box = crop(frame, preprocess, 224, True, device=device)

        if check:
            with torch.no_grad():
                _, pred_val, pred_aro, _, _ = model(output_tensor)
                val = np.clip(pred_val.item(), -1, 1)
                aro = np.clip(pred_aro.item(), -1, 1)

            if prev_val is not None:
                delta_val = abs(val - prev_val)
                delta_aro = abs(aro - prev_aro)
                val_diffs.append(delta_val)
                aro_diffs.append(delta_aro)

            prev_val, prev_aro = val, aro

        frame_idx += 1

    cap.release()

    if len(val_diffs) == 0:
        print("⚠️ 얼굴이 감지되지 않아 변화량 계산 불가.")
        return None, None

    mean_val_change = np.mean(val_diffs)
    mean_aro_change = np.mean(aro_diffs)

    print(f"\n평균 Valence 변화량: {mean_val_change:.4f}")
    print(f"평균 Arousal 변화량: {mean_aro_change:.4f}")

    return mean_val_change, mean_aro_change


video_path = '/home/face/Desktop/LangAgent/langchain_demo.mp4'
val_avg, aro_avg = compute_va_change_average(video_path)