# Facial Recognition Application

This project is a facial recognition application trained on the CelebA dataset using a Siamese Network. The model training was conducted on an NVIDIA A100 GPU using Google Colab, and the inference is executed locally on Apple Silicon (M1/M2) devices. The application uses PyTorch for deep learning, OpenCV for camera integration, and PyQt5 for building the graphical user interface (GUI).

## Project Overview

### Key Features:

- **Model**: A Siamese Network trained for facial recognition tasks.
- **Dataset**: CelebA dataset, processed for one-shot learning.
- **Training**: Conducted on NVIDIA A100 GPU in Colab for efficient computation.
- **Inference**: Optimized for local Apple Silicon using PyTorch's Metal Performance Shaders (MPS) backend.
- **GUI**: Built with PyQt5 to facilitate drag-and-drop and camera input functionality.
- **Technologies**:
  - PyTorch
  - torchvision torchmetrics 
  - OpenCV
  - PyQt5

## Application Structure

### File Descriptions:

1. `app.py`: Contains the PyQt5-based GUI and the logic for handling user interactions and invoking the inference pipeline.
2. `inference.py`: Implements the functions for loading the trained model and performing face recognition.
3. `celeba_facerecognition.ipynb`: Notebook used for training the Siamese Network on the CelebA dataset.
4. `environment.yaml`: Environment file to create a reproducible Conda environment for running the application.

### Architecture:

- **Model Architecture**: Siamese Network consisting of convolutional layers followed by fully connected layers to compute feature embeddings.
- **Loss Function**: Contrastive loss for optimizing the distance between positive and negative pairs.
- **Inference Pipeline**: Resizes and normalizes images, generates embeddings for the anchor and candidate images, and computes similarity scores.

## Requirements

### Hardware:

- NVIDIA A100 GPU (for training).
- Apple Silicon (for local inference).

### Software:

- Python 3.9+
- Conda (for environment management)

## Setup Instructions

### Step 1: Clone the Repository

```bash
# Clone the repository to your local machine
git clone https://github.com/hx88xn/facialrecognition_celeba_triplets.git
cd facialrecognition
```

### Step 2: Create the Conda Environment

Use the provided `environment.yaml` file to create a Conda environment with all necessary dependencies:

```bash
conda env create -f environment.yaml
conda activate facialrec
```

### Step 3: Run the Application

Execute the application:

```bash
python app.py
```

## How to Use the Application

### Main Functionalities:

1. **Drag-and-Drop**: Drag and drop images into the respective fields for anchor and candidate.
2. **Image Selection**: Use file selection dialogs to choose anchor and candidate images.
3. **Live Preview**: Capture real-time images from your camera for anchor and candidate inputs.
4. **Recognition**: Click the "Recognize Face" button to perform face recognition.

### Steps:

1. Start the application using the instructions above.
2. Drag/drop or select images for both anchor and candidate fields.
3. Optionally, use the live preview functionality to take real-time pictures.
4. Click "Recognize Face" to view the result.

## Training Details

### Dataset Preparation:

- CelebA dataset was used for training the model.
- Data preprocessing includes resizing to 128x128, normalization, and creation of positive/negative pairs.

### Training Environment:

- Google Colab with NVIDIA A100 GPU.
- Batch size: 128.
- Loss function: Contrastive loss.
- Optimizer: Adam.

## Acknowledgments

- **CelebA Triplets Dataset on Kaggle**: https://www.kaggle.com/datasets/quadeer15sh/celeba-face-recognition-triplets
- **PyTorch**: [PyTorch Official Documentation](https://pytorch.org/docs/)
- **OpenCV**: [OpenCV Official Documentation](https://opencv.org/)
- **PyQt5**: [PyQt5 Documentation](https://www.riverbankcomputing.com/software/pyqt/intro)
