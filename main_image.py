# 전체 코드 (퍼센트 색상 채움 반영)

import sys
import cv2
import json
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QProgressBar, QMessageBox, QPlainTextEdit, QSizePolicy
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.color_map = {
            'Surprise': '#ffcc00', 'Fear': '#9b59b6', 'Disgust': '#27ae60',
            'Happiness': '#f39c12', 'Sadness': '#3498db', 'Anger': '#e74c3c',
            'Neutral': '#95a5a6', 'No Face': '#7f8c8d'
        }

        self.setStyleSheet("""
            QWidget { background-color: #f2f2f7; font-size: 15px; font-family: 'Helvetica Neue'; }
            QPushButton { background-color: #007aff; color: white; padding: 10px; border-radius: 8px; }
            QPushButton:hover { background-color: #005ecb; }
            QLabel { color: #1c1c1e; }
            QProgressBar {
                border: 1px solid #d1d1d6; border-radius: 5px;
                height: 16px; background-color: #e5e5ea;
                margin-right: 8px;
            }
        """)

        args = get_args_parser()
        args.transform = get_transform()
        args.weights_path = '/home/face/Desktop/NLA/EXP_module/weights/best.pth'
        self.model = NLA_r18(args)
        self.model.load_state_dict(torch.load(args.weights_path, map_location=self.device))
        self.model.to(self.device).eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.577, 0.4494, 0.4001], std=[0.2628, 0.2395, 0.2383])
        ])
        self.exp_dict = {
            0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happiness',
            4: 'Sadness', 5: 'Anger', 6: 'Neutral'
        }

        # 왼쪽 영상
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("background-color: #dcdcdc; border: 1px solid #aaa;")
        self.original_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 우측 상단 미리보기
        self.small_orig_label = QLabel()
        self.small_crop_label = QLabel()
        for l in [self.small_orig_label, self.small_crop_label]:
            l.setFixedSize(160, 160)
            l.setStyleSheet("background-color: white; border: 1px solid #aaa;")

        preview_layout = QHBoxLayout()
        preview_layout.addWidget(self.small_orig_label)
        preview_layout.addWidget(self.small_crop_label)

        # 예측 결과 바
        self.result_label = QLabel("<i>Prediction results will be displayed here.</i>")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px;")

        self.progress_bars = {}
        self.percentage_labels = {}
        self.graph_layout = QVBoxLayout()
        for emotion in self.exp_dict.values():
            self.add_progress_bar(emotion)
        self.add_progress_bar('No Face', color=self.color_map['No Face'])

        self.json_output = QPlainTextEdit()
        self.json_output.setFixedHeight(160)
        self.json_output.setReadOnly(True)

        self.save_btn = QPushButton("Save JSON")
        self.save_btn.clicked.connect(self.save_json)

        # 오른쪽 전체 결과 레이아웃
        right_vbox = QVBoxLayout()
        title = QLabel("RESULT", alignment=Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 18px; background-color: #d1d1d6; padding: 6px;")
        title.setFixedHeight(40)
        right_vbox.addWidget(title)
        right_vbox.addLayout(preview_layout)
        right_vbox.addSpacing(10)
        right_vbox.addLayout(self.graph_layout)
        right_vbox.addWidget(self.result_label)
        right_vbox.addWidget(self.json_output)
        right_vbox.addWidget(self.save_btn)
        right_vbox.addStretch()

        # 버튼
        self.btn = QPushButton("Open Image")
        self.btn.clicked.connect(self.load_image)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn)

        # 좌우 메인 영역
        hbox = QHBoxLayout()
        hbox.addWidget(self.original_label, 2)
        hbox.addLayout(right_vbox, 3)
        hbox.setContentsMargins(10, 10, 10, 10)
        hbox.setSpacing(12)

        layout = QVBoxLayout()
        layout.addLayout(button_layout)
        layout.addLayout(hbox)
        self.setLayout(layout)

        self.reset_ui()

    def add_progress_bar(self, emotion, color=None):
        hbox = QHBoxLayout()
        label = QLabel(f"{emotion}: ")
        label.setFixedWidth(90)
        bar = QProgressBar()
        bar.setMaximum(100)
        bar.setTextVisible(False)
        bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {color or self.color_map.get(emotion, '#ccc')}; border-radius: 5px; }}"
        )
        percent = QLabel("0.00%")
        percent.setFixedWidth(60)
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
        for lbl in self.percentage_labels.values():
            lbl.setText("0.00%")

    def save_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", "expression_result.json", "JSON Files (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.json_output.toPlainText())

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image files (*.jpg *.png)')
        if not path:
            return
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        self.process_frame(image)

    def process_frame(self, image):
        qimg = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.original_label.setPixmap(pixmap.scaled(self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.small_orig_label.setPixmap(pixmap.scaled(self.small_orig_label.size(), Qt.KeepAspectRatio))

        output_tensor, check, box = crop_qt(image, self.preprocess, 224, True, device=self.device)
        if not check:
            self.reset_ui()
            self.progress_bars['No Face'].setValue(100)
            self.percentage_labels['No Face'].setText("100.00%")
            self.result_label.setText("Prediction: No Face")
            self.json_output.setPlainText(json.dumps({"predicted_label": "No Face"}, indent=4))
            return

        with torch.no_grad():
            output = self.model(output_tensor)
            prob = Softmax(dim=1)(output)[0].cpu().numpy()
            prob /= prob.sum()
            pred_cls = np.argmax(prob)
            label = self.exp_dict[pred_cls]

        x1, y1, x2, y2 = map(int, box)
        crop_img = image[y1:y2, x1:x2]
        crop_img = cv2.resize(crop_img, (160, 160))
        qimg_crop = QImage(crop_img.data, crop_img.shape[1], crop_img.shape[0], crop_img.strides[0], QImage.Format_RGB888)
        self.small_crop_label.setPixmap(QPixmap.fromImage(qimg_crop).scaled(self.small_crop_label.size(), Qt.KeepAspectRatio))

        self.result_label.setText(f"<b>Prediction: <span style='color:#ff9500'>{label}</span> ({prob[pred_cls]*100:.2f}%)</b>")
        for emo, val in zip(self.exp_dict.values(), prob):
            percent = round(val * 100, 2)
            self.progress_bars[emo].setValue(int(percent))
            self.percentage_labels[emo].setText(f"{percent:.2f}%")
        self.progress_bars['No Face'].setValue(0)
        self.percentage_labels['No Face'].setText("0.00%")

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







