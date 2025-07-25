# Emotion-Recognizer-AI
# Emotion AI: Facial Keypoint & Emotion Detection

## 📌 Overview

Emotion Recognizer AI, or Artificial Emotional Intelligence, is a branch of AI focused on understanding human emotions using non-verbal cues like facial expressions and body language. This project uses deep learning to perform:

1. Facial Keypoint Detection – Predicting coordinates of important facial landmarks.
2. Facial Emotion Classification – Classifying facial expressions into discrete emotion categories.

---

## 🧠 Project Objectives

- Build a system that detects facial keypoints and classifies emotions from facial images.
- Train two separate models using deep learning: one for landmark detection, another for emotion classification.
- Deploy models using TensorFlow Serving for real-time inference.

---

## 🗂️ Datasets

### 🔹 Facial Keypoint Detection
- Source: [Kaggle - Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection/data)
- Images: 96x96 grayscale
- Targets: X, Y coordinates of 15 facial keypoints

### 🔹 Facial Expression Recognition
- Source: [Kaggle - Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- Images: 48x48 grayscale
- Classes:
  - 0: Angry
  - 1: Disgust
  - 2: Sad
  - 3: Happy
  - 4: Surprise

---

## 🧪 Technologies Used

| Category               | Tools & Libraries                                                                 |
|------------------------|------------------------------------------------------------------------------------|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow, Keras |
| Model Architecture | CNN, ResNet (Residual Networks) |
| Deployment | TensorFlow Serving |
| Request Handling | Python requests library (for RESTful API calls) |
| Loss Function | Mean Squared Error (for regression), Cross Entropy (for classification) |
| Optimizer | Gradient Descent / Adam |
| Visualization | Confusion Matrix, Precision/Recall, Accuracy |

---

## 🧰 Model Architecture

### 📍 Keypoint Detection Model
- CNN with Residual Blocks
- Output: 30 values (15 (x, y) coordinates)

### 😊 Emotion Classification Model
- Deep CNN
- Output: Softmax activation for 5 emotion classes

### 🧱 ResNet Highlights
- Solves vanishing gradient problem in deep networks
- Includes skip connections and identity blocks

---

## 🚀 Model Training & Evaluation

- Data split: 60% training / 20% validation / 20% test
- Training set used for weight updates
- Validation set to prevent overfitting
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix

---

## 🌐 Model Deployment

### 🔧 Using TensorFlow Serving

Steps to deploy:
1. Export model in TF Serving-compatible format.
2. Assign a model name and version directory.
3. Run TF Serving with:
   ```bash
   tensorflow_model_server \
     --rest_api_port=8501 \
     --model_name=emotion_ai \
     --model_base_path="/models/emotion_ai/"
