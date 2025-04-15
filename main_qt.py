# 전체 화면 확대, 결과 우측 상단에 원본 + 크롭 이미지 작게 표시
import sys
import cv2
import json
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QProgressBar, QMessageBox, QPlainTextEdit, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from torchvision import transforms
from torch.nn import Softmax

from EXP_module.src.model import NLA_r18
from EXP_module.src.utils import *
from face_det_module.src.util import get_args_parser, get_transform
from face_det_module.src.face_crop import crop_qt

class ExpressionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Expression Recognition")
        self.showMaximized()

        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.color_map = {
            'Surprise': '#ffcc00',
            'Fear': '#9b59b6',
            'Disgust': '#27ae60',
            'Happiness': '#f39c12',
            'Sadness': '#3498db',
            'Anger': '#e74c3c',
            'Neutral': '#95a5a6'
        }

        # PyQt 스타일
        self.setStyleSheet("""
            QWidget { background-color: #f2f2f7; font-family: 'Helvetica Neue'; font-size: 14px; }
            QPushButton { background-color: #007aff; color: white; padding: 10px; border-radius: 10px; }
            QPushButton:hover { background-color: #005ecb; }
            QLabel { color: #1c1c1e; }
            QProgressBar { border: 1px solid #d1d1d6; border-radius: 5px; height: 16px; background-color: #e5e5ea; margin-right: 8px; text-align: center; }
        """)

        args = get_args_parser()
        args.transform = get_transform()
        args.weights_path = '/home/face/Desktop/NLA/EXP_module/weights/best.pth'
        self.model = NLA_r18(args)
        self.model.load_state_dict(torch.load(args.weights_path, map_location=self.device))
        self.model = self.model.to(self.device).eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.577, 0.4494, 0.4001],
                                 std=[0.2628, 0.2395, 0.2383])
        ])

        self.exp_dict = {0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happiness', 4: 'Sadness', 5: 'Anger', 6: 'Neutral'}

        # 왼쪽 원본 이미지
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("background-color: #dcdcdc; border: 1px solid #aaa;")
        self.original_label.setScaledContents(False)

        # 오른쪽 상단에 작게 표시할 원본/크롭 이미지
        self.small_orig_label = QLabel()
        self.small_crop_label = QLabel()
        for label in [self.small_orig_label, self.small_crop_label]:
            label.setFixedSize(160, 160)
            label.setStyleSheet("border: 1px solid #ccc; background-color: white;")

        small_imgs_layout = QHBoxLayout()
        small_imgs_layout.addWidget(self.small_orig_label)
        small_imgs_layout.addWidget(self.small_crop_label)

        # 결과 타이틀
        result_title = QLabel("RESULT")
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("font-weight: bold; font-size: 18px; background-color: #d1d1d6; padding: 8px; margin-bottom: 6px; min-height: 30px;")

        # 결과 라벨
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; margin-top: 12px; color: #1c1c1e;")

        # Progress Bar
        self.progress_bars = {}
        self.percentage_labels = {}
        self.graph_layout = QVBoxLayout()
        for idx, emotion in self.exp_dict.items():
            self.add_progress_bar(emotion)
        self.add_progress_bar('No Face', color='#7f8c8d')

        # JSON 결과
        self.json_output = QPlainTextEdit()
        self.json_output.setFixedHeight(170)
        self.json_output.setReadOnly(True)

        self.save_btn = QPushButton("Save JSON")
        self.save_btn.clicked.connect(self.save_json)
        self.save_btn.setFixedWidth(120)

        # 오른쪽 전체 구성
        right_vbox = QVBoxLayout()
        right_vbox.addWidget(result_title)
        right_vbox.addLayout(small_imgs_layout)
        right_vbox.addSpacing(10)
        right_vbox.addLayout(self.graph_layout)
        right_vbox.addWidget(self.result_label)
        right_vbox.addWidget(self.json_output)
        right_vbox.addWidget(self.save_btn)
        right_vbox.addStretch()

        # 이미지 열기 버튼
        self.btn = QPushButton("Open Image")
        self.btn.clicked.connect(self.load_image)

        top_vbox = QVBoxLayout()
        top_vbox.addWidget(self.btn)

        hbox = QHBoxLayout()
        hbox.addWidget(self.original_label, 2)
        hbox.addLayout(right_vbox, 3)

        layout = QVBoxLayout()
        layout.addLayout(top_vbox)
        layout.addLayout(hbox)
        self.setLayout(layout)

        self.reset_ui()

    def reset_ui(self):
        self.original_label.clear()
        self.small_orig_label.clear()
        self.small_crop_label.clear()
        self.result_label.setText("<i>Prediction results will be displayed here.</i>")
        self.json_output.clear()
        for emotion in self.exp_dict.values():
            self.progress_bars[emotion].setValue(0)
            self.percentage_labels[emotion].setText("0.00%")

    def save_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", "expression_result.json", "JSON Files (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.json_output.toPlainText())

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image files (*.jpg *.png)')
        if not path:
            return
        bgr_img = cv2.imread(path)
        image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # 원본 전체 표시
        qimg_orig = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg_orig)
        self.original_label.setPixmap(pixmap.scaled(self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.small_orig_label.setPixmap(pixmap.scaled(self.small_orig_label.size(), Qt.KeepAspectRatio))

        # 얼굴 검출 및 예측
        output_tensor, check, box = crop_qt(image, self.preprocess, 224, True, device=self.device)
        if not check:
            self.reset_ui()
            self.original_label.setPixmap(pixmap.scaled(self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            return

        with torch.no_grad():
            output = self.model(output_tensor)
            prob = Softmax(dim=1)(output)
            prob = prob.squeeze(0).cpu().numpy()
            prob = prob / prob.sum()
            pred_cls = np.argmax(prob)
            label = self.exp_dict[pred_cls]

        # 얼굴 crop된 이미지 표시
        x1, y1, x2, y2 = map(int, box)
        crop_img = image[y1:y2, x1:x2]
        crop_img = cv2.resize(crop_img, (160, 160))
        qimg_crop = QImage(crop_img.data, crop_img.shape[1], crop_img.shape[0], crop_img.strides[0], QImage.Format_RGB888)
        self.small_crop_label.setPixmap(QPixmap.fromImage(qimg_crop).scaled(self.small_crop_label.size(), Qt.KeepAspectRatio))

        for emo_label, value in zip(self.exp_dict.values(), prob):
            percent = round(float(value) * 100, 2)
            self.progress_bars[emo_label].setValue(int(percent))
            self.percentage_labels[emo_label].setText(f"{percent:.2f}%")

        self.result_label.setText(f"<b>Prediction: <span style='color:#ff9500'>{label}</span> ({prob[pred_cls]*100:.2f}%)</b>")
        result_json = {
            "predicted_label": label,
            "predicted_index": int(pred_cls),
            "softmax": {k: float(f"{v:.4f}") for k, v in zip(self.exp_dict.values(), prob)},
            "image_size": {"width": image.shape[1], "height": image.shape[0]},
            "face_box": {"x1": int(box[0]), "y1": int(box[1]), "x2": int(box[2]), "y2": int(box[3])}
        }
        self.json_output.setPlainText(json.dumps(result_json, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ExpressionApp()
    ex.show()
    sys.exit(app.exec_())











