<h1 align="center"><b>😷 face-mask-scanner 😷</b></h1>


## 📌 Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training Details](#training-details)
5. [Live Detection](#live-detection)
6. [Results](#results)
7. [How to Run](#how-to-run)

---

## 📖 Introduction

Face Mask Scanner is a real-time face mask detection system that identifies whether a person is wearing a face mask or not using deep learning. The system is capable of processing both image inputs and live video streams.

The core objective of this project is:
> "To alert authorities if someone is not wearing a face mask, helping enforce safety in public places especially during pandemics like COVID-19."

---

## 📂 Dataset

- **Dataset Used:** [COVID Face Mask Detection Dataset](https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset)
- **Classes:**
  - With Mask 😷
  - Without Mask 😮
- **Total Images:** ~7,500 (Training + Validation)

---

## 🧠 Model Architecture

The model is based on **MobileNetV2**, a lightweight convolutional neural network designed for efficient on-device vision applications. Key components:

- **Base Model:** MobileNetV2 (pre-trained on ImageNet, frozen during initial training)
- **Custom Head:**
  - AveragePooling2D
  - Flatten → Dense(128, ReLU)
  - Dropout(0.5)
  - Dense(2, Softmax)

---

## 🏋️‍♀️ Training Details

- **Image Size:** 224x224
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam (lr=1e-4)
- **Epochs:** 50
- **Batch Size:** 32
- **Data Augmentation:** Rotation, Zoom, Shift, Shear, Flip
- **Accuracy Achieved:** **~98.32% on Validation Set**

---

## 🎥 Live Detection

- Real-time face mask detection from webcam using **OpenCV**.
- Utilizes the trained MobileNetV2 model (`face_mask_model.h5`) for inference.

> Run `live-detection.py` to start webcam-based mask detection.

---

## ✅ Results

| Metric         | Value     |
|----------------|-----------|
| Accuracy       | 98.32%    |
| Precision      | 97.85%    |
| Recall         | 98.67%    |
| Model Size     | ~14 MB    |
| Framework Used | TensorFlow / Keras |

---

## 🚀 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/face-mask-scanner.git
cd face-mask-scanner
