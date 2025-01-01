import torch
from torch import nn
from torchvision import transforms, models
from facenet_pytorch import MTCNN
from PIL import Image
import cv2

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)

    def forward_once(self, x):
        return self.resnet(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

def detect_and_preprocess_face(image_path, transform):
    detector = MTCNN()
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        boxes, _ = detector.detect(img_rgb)
        if boxes is not None:
            x1, y1, x2, y2 = [int(coord) for coord in boxes[0]]
            face = img_rgb[y1:y2, x1:x2]
            face = Image.fromarray(face).resize((128, 128))
            return transform(face)
        else:
            raise ValueError("No face detected in the image!")
    except Exception as e:
        raise ValueError(f"Error in face detection: {e}")

def recognize_face(model, anchor_image, candidate_image, transform, threshold=0.49):
    model.eval()

    device = next(model.parameters()).device

    try:
        anchor = detect_and_preprocess_face(anchor_image, transform).unsqueeze(0).to(device)
        candidate = detect_and_preprocess_face(candidate_image, transform).unsqueeze(0).to(device)
    except ValueError as e:
        return str(e)

    with torch.no_grad():
        anchor_embedding, candidate_embedding = model(anchor, candidate)
        distance = torch.sqrt(torch.sum((anchor_embedding - candidate_embedding) ** 2, dim=1)).item()

    if distance <= threshold:
        return f"Face recognized. Similarity score: {1 - distance:.2f}"
    else:
        return f"Face not recognized. Similarity score: {1 - distance:.2f}"

def load_model(model_path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = SiameseNetwork().to(device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
