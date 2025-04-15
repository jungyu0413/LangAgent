import cv2
import numpy as np
import torch
from torchvision import transforms
from VA_module.src.resnet import VA_Model
from face_det_module.src.util import get_args_parser, get_transform, pre_trained_wegiths_load
from face_det_module.src.face_crop import crop
from face_det_module.src.util import resize_image

# Font & drawing settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
v_font_color = (0, 255, 0)
a_font_color = (0, 0, 255)
font_thickness = 2
resize_param = 250

if __name__ == "__main__":
    # Load model
    args = get_args_parser()
    args.transform = get_transform()
    args.weights_path = '/home/face/Desktop/LangAgent/VA_module/weights/best.pth'

    model = VA_Model(args)
    cp = torch.load(args.weights_path, map_location='cuda')
    model = pre_trained_wegiths_load(model, cp)
    model = model.to('cuda').eval()

    # Load VA map
    VA_img = cv2.imread('/home/face/Desktop/LangAgent/VA_module/VA.jpg')
    if VA_img is None:
        raise FileNotFoundError("VA.jpg 파일을 찾을 수 없습니다.")
    
    VA, center_info = resize_image(VA_img, resize_param)
    new_center_h, new_center_w, new_length = center_info
    va_h, va_w = VA.shape[:2]

    # Drawing setting
    radius = 2
    color = (0, 0, 255)
    thickness = -1

    # Start camera
    cap = cv2.VideoCapture(0)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.577, 0.4494, 0.4001],
                             std=[0.2628, 0.2395, 0.2383])
    ])

    while True:
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.flip(image, 1)
        img_h, img_w = image.shape[:2]
        output_tensor, check, box = crop(image, preprocess, 224, True, device='cuda')

        # 기본값
        map_val, map_aro = 0, 0

        if check:
            with torch.no_grad():
                _, pred_val, pred_aro, _, _ = model(output_tensor)
                valence = np.clip(np.round(pred_val.item(), 2), -1, 1)
                arousal = np.clip(np.round(pred_aro.item(), 2), -1, 1)
                map_val = int(valence * (new_length / 2))
                map_aro = int(arousal * (new_length / 2))

            cv2.putText(image, f"Valence: {valence}", (10, 50), font, font_size, v_font_color, font_thickness)
            cv2.putText(image, f"Arousal: {arousal}", (10, 100), font, font_size, a_font_color, font_thickness)
        else:
            cv2.putText(image, "Valence: None", (10, 50), font, font_size, v_font_color, font_thickness)
            cv2.putText(image, "Arousal: None", (10, 100), font, font_size, a_font_color, font_thickness)

        # 오버레이: VA 맵
        image[img_h - va_h:, img_w - va_w:] = VA
        center_h = img_h - (va_h // 2)
        center_w = img_w - (va_w // 2)
        cv2.circle(image, (center_w + map_val, center_h - map_aro), radius, color, thickness)

        cv2.imshow("Output", image)
        k = cv2.waitKey(2) & 0xFF
        if k == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
