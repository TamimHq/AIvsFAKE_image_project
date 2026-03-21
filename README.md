# 🧠 AI vs Real Image Classification using CNN

## 📌 Overview
This project develops a deep learning model to classify whether an image is **real or AI-generated** using Convolutional Neural Networks (CNN). With the rapid rise of generative AI, detecting synthetic images has become a critical challenge, and this project aims to address that problem.

---

## 🎯 Problem Statement
The advancement of generative AI models has made it increasingly difficult to distinguish between real and AI-generated images. This project builds an automated system to classify images into two categories:
- Real Images
- AI-Generated Images

---

## 📂 Dataset
- Dataset: CIFAKE (Kaggle)
- Source: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- Structure:
  - `train/REAL`
  - `train/FAKE`
  - `test/REAL`
  - `test/FAKE`

---

## 🧠 Methodology

### 🔹 Data Preprocessing
- Rescaled images (pixel normalization: 1/255)
- Resized images to 128x128
- Applied data augmentation:
 - Rotation
 - Zoom
 - Shift (width & height)
 - Brightness adjustment
 - Horizontal flip
- Used validation split (20%) for better generalization

### 🔹 Model Architecture
- Convolutional Neural Network (CNN)
  - Conv2D (32) → BatchNorm → MaxPooling
  - Conv2D (64) → BatchNorm → MaxPooling
  - Conv2D (128) → BatchNorm → MaxPooling
  - Conv2D (256) → BatchNorm → MaxPooling
  - GlobalAveragePooling
  - Dense (128)
  - Dropout (0.5)
  - Output Layer (Sigmoid)

### 🔹 Training
- Loss Function: Binary Crossentropy
- Optimizer: Adam (lr = 1e-4)
- Epochs: 20
- Batch Size: 32
- Validation: Separate validation set (not test data)
**Callbacks Used:**
- EarlyStopping (prevents overfitting)
- ReduceLROnPlateau (adaptive learning rate)

---

## 📊 Results
- Model evaluated on test dataset
- Achieved: **94.35% accuracy (on CIFAKE dataset)**
- Performance analyzed using:
  - Training vs Validation Accuracy Graph
  - Confusion Matrix
  - Test Accuracy & Loss

---

## 📈 Visualizations
### 🔹 Training vs Validation Accuracy
<p align="center">
  <img src="results/accuracy.png" width="500"/>
</p>

### 🔹 Confusion Matrix
<p align="center">
  <img src="results/confusion_matrix.png" width="500"/>
</p>

### 🔹 Test Metrics
<p align="center">
  <img src="results/test_metrics.png" width="500"/>
</p>

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## ▶️ How to Run
### 1. Kaggle Setup
Download `kaggle.json` from your Kaggle account.
 ```python
   !mkdir -p ~/.kaggle
   !mv kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```
### 2. Download Dataset
 ```python
   !kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
   !unzip cifake-real-and-ai-generated-synthetic-images.zip
   ```
### 3. Run the notebook 
Open and run: 
```bash
ai_vs_real_image_classification.ipynb
```
---
## 🔍 Prediction Example
```python
   The image is predicted to be: Real
```
---
## 🧪 Model Inference (Sample Function)
```python
def predict_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return "Real" if prediction[0][0] > 0.5 else "AI-Generated"
```
---
## ⚠️ Limitations
- Model performs well on CIFAKE dataset but may struggle on:
 - Real-world camera images
 - Social media images
 - AI images from unseen generators

This is due to **dataset bias and domain shift**.

---
## 🔧 Future Improvements
- Use Transfer Learning (EfficientNet / MobileNet)
- Increase image size (e.g., 224×224)
- Train on more diverse datasets
- Improve real-world generalization
- Add deployment (web/app interface)
---
## 📁 Project Structure
```bash
AIvsFAKE_image_project/
├── AI_VS_FAKE.ipynb
├── README.md
├── LICENSE
└── results/
    ├── accuracy.png
    ├── confusion_matrix.png
    └── test_metrics.png
```
```md
This project demonstrates a complete deep learning pipeline from data preprocessing to model evaluation and inference.
