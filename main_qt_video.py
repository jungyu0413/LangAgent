import sys
import cv2
import json
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QProgressBar, QMessageBox,
                             QPlainTextEdit, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
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
        self.resize(1400, 800)

        # Device (GPU/CPU) 설정
        use_cuda = True
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        # 감정별 색상
        self.color_map = {
            'Surprise': '#ffcc00', 'Fear': '#9b59b6', 'Disgust': '#27ae60',
            'Happiness': '#f39c12', 'Sadness': '#3498db', 'Anger': '#e74c3c', 'Neutral': '#95a5a6'
        }

        # PyQt5 전체 스타일
        self.setStyleSheet("""
            QWidget { background-color: #f2f2f7; font-family: -apple-system, 'Helvetica Neue', sans-serif; font-size: 14px; }
            QPushButton { background-color: #007aff; color: white; padding: 8px; border-radius: 8px; }
            QPushButton:hover { background-color: #005ecb; }
            QLabel { color: #1c1c1e; }
            QProgressBar { border: 1px solid #d1d1d6; border-radius: 5px; height: 18px; background-color: #e5e5ea; margin-right: 6px; text-align: center; }
        """)

        # 모델 불러오기
        args = get_args_parser()
        args.transform = get_transform()
        args.weights_path = '/home/face/Desktop/NLA/EXP_module/weights/best.pth'
        self.model = NLA_r18(args)
        self.model.load_state_dict(torch.load(args.weights_path, map_location=self.device))
        self.model = self.model.to(self.device).eval()

        # 이미지 전처리
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.577, 0.4494, 0.4001],
                                 std=[0.2628, 0.2395, 0.2383])
        ])

        # 클래스 딕셔너리
        self.exp_dict = {0: 'Surprise', 1: 'Fear', 2: 'Disgust',
                         3: 'Happiness', 4: 'Sadness', 5: 'Anger', 6: 'Neutral'}

        # 왼쪽 원본 이미지 표시 QLabel
        self.original_label = QLabel(self)
        self.original_label.setFixedSize(500, 700)  # <<< 고정
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("background-color: #dcdcdc; border: 1px solid #aaa;")

        # 오른쪽 (작은 원본/크롭 이미지) QLabel
        self.small_orig_label = QLabel(self)
        self.small_crop_label = QLabel(self)
        for label in [self.small_orig_label, self.small_crop_label]:
            label.setFixedSize(160, 160)
            label.setStyleSheet("border: 1px solid #ccc; background-color: white;")

        # 예측 결과 레이블
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; margin-top: 10px; color: #1c1c1e;")

        # softmax 확률 표시 ProgressBar
        self.progress_bars = {}
        self.percentage_labels = {}
        self.graph_layout = QVBoxLayout()
        for emotion in self.exp_dict.values():
            self.add_progress_bar(emotion)
        self.add_progress_bar('No Face', color='#7f8c8d')

        # JSON 결과 표시창
        self.json_output = QPlainTextEdit()
        self.json_output.setFixedHeight(150)
        self.json_output.setReadOnly(True)
        self.json_output.setStyleSheet("background-color: white; border: 1px solid #ccc; padding: 6px;")

        # JSON 저장 버튼
        self.save_btn = QPushButton("Save JSON")
        self.save_btn.setFixedWidth(150)
        self.save_btn.clicked.connect(self.save_json)

        # 파일 열기 버튼
        self.btn_image = QPushButton('Open Image')
        self.btn_image.setFixedWidth(500)
        self.btn_image.clicked.connect(self.load_image)

        self.btn_video = QPushButton('Open Video')
        self.btn_video.setFixedWidth(880)
        self.btn_video.clicked.connect(self.load_video)

        # 레이아웃 설정
        top_buttons = QHBoxLayout()
        top_buttons.addWidget(self.btn_image)
        top_buttons.addWidget(self.btn_video)

        top_images_hbox = QHBoxLayout()
        top_images_hbox.addWidget(self.small_orig_label)
        top_images_hbox.addWidget(self.small_crop_label)

        right_vbox = QVBoxLayout()
        self.result_title = QLabel("RESULT", alignment=Qt.AlignCenter)
        self.result_title.setFixedHeight(40)
        self.result_title.setFixedWidth(880)
        self.result_title.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #d1d1d6; padding: 5px;")

        right_vbox.addWidget(self.result_title)
        right_vbox.addLayout(top_images_hbox)
        right_vbox.addSpacing(10)
        right_vbox.addLayout(self.graph_layout)
        right_vbox.addWidget(self.result_label)
        right_vbox.addWidget(self.json_output)
        right_vbox.addWidget(self.save_btn)
        right_vbox.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        main_hbox = QHBoxLayout()
        main_hbox.addWidget(self.original_label)
        main_hbox.addLayout(right_vbox)

        layout = QVBoxLayout()
        layout.addLayout(top_buttons)
        layout.addLayout(main_hbox)
        self.setLayout(layout)

        # 상태 변수
        self.video_cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_video_frame)
        self.missing_face_counter = 0
        self.last_prediction = None

        self.reset_ui()

    def add_progress_bar(self, emotion, color=None):
        hbox = QHBoxLayout()
        label = QLabel(f"{emotion}: ")
        label.setFixedWidth(90)
        label.setStyleSheet("font-weight: bold; color: #3a3a3c;")
        bar = QProgressBar()
        bar.setMaximum(100)
        bar.setTextVisible(False)
        color_code = color if color else self.color_map.get(emotion, '#ccc')
        bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color_code}; border-radius: 5px; }}")
        percent = QLabel("0.00%")
        percent.setFixedWidth(60)
        percent.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        hbox.addWidget(label)
        hbox.addWidget(bar)
        hbox.addWidget(percent)
        self.progress_bars[emotion] = bar
        self.percentage_labels[emotion] = percent
        self.graph_layout.addLayout(hbox)

    def reset_ui(self):
        self.original_label.clear()
        self.small_orig_label.clear()
        self.small_crop_label.clear()
        self.result_label.setText("<i>Prediction results will be displayed here.</i>")
        self.json_output.clear()
        for bar in self.progress_bars.values():
            bar.setValue(0)
        for label in self.percentage_labels.values():
            label.setText("0.00%")

    def save_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", "expression_result.json", "JSON Files (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.json_output.toPlainText())

    def load_image(self):
        self.timer.stop()
        path, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image files (*.jpg *.png)')
        if path:
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            self.process_frame(image)

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Video', '', 'Video files (*.mp4 *.avi)')
        if path:
            self.video_cap = cv2.VideoCapture(path)
            self.timer.start(30)

    def process_video_frame(self):
        if self.video_cap is None or not self.video_cap.isOpened():
            self.timer.stop()
            return
        ret, frame = self.video_cap.read()
        if not ret:
            self.timer.stop()
            return
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.process_frame(image, video_mode=True)

    def process_frame(self, image, video_mode=False):
        # 원본 표시 (항상 고정 크기)
        pixmap = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888))
        self.original_label.setPixmap(pixmap.scaled(self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        output_tensor, check, box = crop_qt(image, self.preprocess, 224, True, device=self.device)
        if not check:
            self.missing_face_counter += 1
            if self.missing_face_counter >= 10:
                self.update_results(None)
            return

        self.missing_face_counter = 0
        with torch.no_grad():
            output = self.model(output_tensor)
            prob = Softmax(dim=1)(output)
            prob = prob.squeeze(0).cpu().numpy()
            prob = prob / prob.sum()
            pred_cls = np.argmax(prob)
            label = self.exp_dict[pred_cls]

        x1, y1, x2, y2 = map(int, box)
        crop_img = image[y1:y2, x1:x2]
        crop_img = cv2.resize(crop_img, (160, 160))
        self.small_crop_label.setPixmap(QPixmap.fromImage(QImage(crop_img.data, crop_img.shape[1], crop_img.shape[0], crop_img.strides[0], QImage.Format_RGB888)))

        self.last_prediction = (label, prob, box, image.shape)
        self.update_results(self.last_prediction)

    def update_results(self, prediction):
        if prediction is None:
            for emo in self.exp_dict.values():
                self.progress_bars[emo].setValue(0)
                self.percentage_labels[emo].setText("0.00%")
            self.progress_bars['No Face'].setValue(100)
            self.percentage_labels['No Face'].setText("100.00%")
            self.result_label.setText(f"<b>Prediction: <span style='color:#7f8c8d'>No Face</span></b>")
            self.json_output.setPlainText(json.dumps({"predicted_label": "No Face", "softmax": {}}, indent=4))
            return

        label, prob, box, img_shape = prediction
        for emo_label, value in zip(self.exp_dict.values(), prob):
            percent = round(float(value) * 100, 2)
            self.progress_bars[emo_label].setValue(int(percent))
            self.percentage_labels[emo_label].setText(f"{percent:.2f}%")
        self.progress_bars['No Face'].setValue(0)
        self.percentage_labels['No Face'].setText("0.00%")

        pred_idx = list(self.exp_dict.values()).index(label)
        self.result_label.setText(
            f"<b>Prediction: <span style='color:#ff9500'>{label}</span> ({prob[pred_idx]*100:.2f}%)</b>"
        )
        result_json = {
            "predicted_label": label,
            "predicted_index": int(pred_idx),
            "softmax": {k: float(f"{v:.4f}") for k, v in zip(self.exp_dict.values(), prob)},
            "image_size": {"width": img_shape[1], "height": img_shape[0]},
            "face_box": {"x1": int(box[0]), "y1": int(box[1]), "x2": int(box[2]), "y2": int(box[3])}
        }
        self.json_output.setPlainText(json.dumps(result_json, indent=4))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ExpressionApp()
    ex.show()
    sys.exit(app.exec_())












