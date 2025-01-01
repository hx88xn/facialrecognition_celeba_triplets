import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import cv2
from inference import load_model, recognize_face
from torchvision import transforms
import torch

class DraggableLabel(QLabel):
    def __init__(self, text='', parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setAcceptDrops(True)  

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): 
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.parent().handle_dropped_image(self, file_path)

class FaceRecognitionApp(QWidget):
    def __init__(self, model, transform):
        super().__init__()
        self.model = model
        self.transform = transform
        self.anchor_image = None
        self.candidate_image = None
        self.captured_images = []  
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Face Recognition")
        self.layout = QVBoxLayout()

        self.anchor_display = DraggableLabel("Drag and drop anchor image here", self)
        self.anchor_display.setAlignment(Qt.AlignCenter)
        self.anchor_display.setStyleSheet("border: 1px solid black;")
        self.anchor_display.setFixedSize(200, 200)
        self.layout.addWidget(self.anchor_display)

        self.select_anchor_btn = QPushButton("Select Anchor Image")
        self.select_anchor_btn.clicked.connect(self.select_anchor_image)
        self.layout.addWidget(self.select_anchor_btn)

        self.anchor_live_preview_btn = QPushButton("Take a picture (Anchor)")
        self.anchor_live_preview_btn.clicked.connect(lambda: self.live_preview("anchor"))
        self.layout.addWidget(self.anchor_live_preview_btn)

        self.candidate_display = DraggableLabel("Drag and drop candidate image here", self)
        self.candidate_display.setAlignment(Qt.AlignCenter)
        self.candidate_display.setStyleSheet("border: 1px solid black;")
        self.candidate_display.setFixedSize(200, 200)
        self.layout.addWidget(self.candidate_display)

        self.select_candidate_btn = QPushButton("Select Candidate Image")
        self.select_candidate_btn.clicked.connect(self.select_candidate_image)
        self.layout.addWidget(self.select_candidate_btn)

        self.candidate_live_preview_btn = QPushButton("Take a picture (Candidate)")
        self.candidate_live_preview_btn.clicked.connect(lambda: self.live_preview("candidate"))
        self.layout.addWidget(self.candidate_live_preview_btn)

        self.recognize_btn = QPushButton("Recognize Face")
        self.recognize_btn.clicked.connect(self.recognize_face)
        self.layout.addWidget(self.recognize_btn)

        self.setLayout(self.layout)

    def select_anchor_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Anchor Image")
        if file_path:
            self.load_image(self.anchor_display, file_path, "anchor")

    def select_candidate_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Candidate Image")
        if file_path:
            self.load_image(self.candidate_display, file_path, "candidate")

    def load_image(self, label, file_path, img_type):
        pixmap = QPixmap(file_path).scaled(200, 200, Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
        if img_type == "anchor":
            self.anchor_image = file_path
        elif img_type == "candidate":
            self.candidate_image = file_path

    def live_preview(self, img_type):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(f"Live Preview ({img_type.capitalize()}) - Press 's' to save, 'q' to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                file_path = f"captured_{img_type}.jpg"
                cv2.imwrite(file_path, frame)
                self.captured_images.append(file_path) 
                if img_type == "anchor":
                    self.load_image(self.anchor_display, file_path, "anchor")
                elif img_type == "candidate":
                    self.load_image(self.candidate_display, file_path, "candidate")
                break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def recognize_face(self):
        if not self.anchor_image or not self.candidate_image:
            QMessageBox.warning(self, "Warning", "Both images must be selected!")
            return

        result = recognize_face(self.model, self.anchor_image, self.candidate_image, self.transform)
        QMessageBox.information(self, "Result", result)

    def handle_dropped_image(self, label, file_path):
        pixmap = QPixmap(file_path).scaled(200, 200, Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
        if label == self.anchor_display:
            self.anchor_image = file_path
        elif label == self.candidate_display:
            self.candidate_image = file_path

    def closeEvent(self, event):
        self.cleanup_images()
        event.accept()

    def cleanup_images(self):
        for image_path in self.captured_images:
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted captured image: {image_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    model_path = 'models/celeba-final.pth'
    print(f"PyTorch version: {torch.__version__}\nMPS available: {torch.backends.mps.is_available()}\nMPS built: {torch.backends.mps.is_built()}")

    #i used 'mps' for inference on apple silicon, you should refer to cuda if your system has support for it for better inference
    device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
    print(f"Using device: {device}")
    model = load_model(model_path, device)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    window = FaceRecognitionApp(model, transform)
    window.show()
    sys.exit(app.exec_())
