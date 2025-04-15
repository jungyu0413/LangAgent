import cv2
import json
import torch
import numpy as np
from torchvision import transforms
from torch.nn import Softmax

from EXP_module.src.model import NLA_r18
from EXP_module.src.utils import *
from face_det_module.src.util import get_args_parser, get_transform
from face_det_module.src.face_crop import crop

def main():
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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("📷 Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 감지 및 crop
        output_tensor, check, box = crop(image_rgb, preprocess, 224, True, device=device)

        if not check:
            label = "No Face"
            prob_dict = {"No Face": 1.0}
        else:
            with torch.no_grad():
                output = model(output_tensor)
                prob = Softmax(dim=1)(output)[0].cpu().numpy()
                prob /= prob.sum()
                pred_cls = np.argmax(prob)
                label = exp_dict[pred_cls]
                prob_dict = {k: float(f"{v:.4f}") for k, v in zip(exp_dict.values(), prob)}

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)

        # 예측 텍스트 그리기
        text = f"{label}"
        cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        # 출력
        print(json.dumps({
            "predicted_label": label,
            "softmax": prob_dict
        }, indent=2, ensure_ascii=False))

        cv2.imshow("Expression Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
