# 🍎 Multilevel Classification CNN Model Project (Fruit Classification)

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-v3-red.svg)](https://keras.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

A modular deep learning project implementing an end-to-end Convolutional Neural Network (CNN) pipeline for multi-class image classification across four fruit categories: **Apple**, **Banana**, **Mango**, and **Orange**.

The project addresses limited training sample challenges through **leakage-free stratified splitting**, **offline data augmentation**, **deep CNN feature extraction**, and **inference visualization**.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [Repository Structure](#-repository-structure)
- [Dataset Specifications](#-dataset-specifications)
- [Data Augmentation Pipeline](#-data-augmentation-pipeline)
- [CNN Architecture & Hyperparameters](#-cnn-architecture--hyperparameters)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [Environment Setup & Installation](#-environment-setup--installation)
- [Execution Guide](#-execution-guide)
- [Prediction & Inference Guide](#-prediction--inference-guide)
- [Technical Highlights & Best Practices](#-technical-highlights--best-practices)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🔍 Project Overview

Accurate visual recognition of agricultural and grocery products is critical for automated inventory tracking, retail checkout automation, and robotic harvesting. This project provides a robust, reproducible computer vision pipeline tailored for multi-category fruit recognition:

1. **Target Classes**: 4 classes (`apple`, `banana`, `mango`, `orange`).
2. **Core Challenge**: High intra-class variance with limited initial training imagery.
3. **Engineered Solution**:
   - Strict separation of validation and test sets *prior* to data generation to prevent data contamination.
   - Offline batch data augmentation scaling the training corpus by 500%.
   - Convolutional neural network regularized with spatial and dense Dropout layers.
   - Comprehensive inference pipeline providing visual confidence and probability distributions.

---

## 🏗️ System Architecture & Workflow

The pipeline is partitioned into three sequential, decoupled Jupyter notebooks:

```
[ dataset/origin/ ]
        │
        ▼ (01_Image_generator.ipynb)
┌─────────────────────────────────────────────────────────┐
│ 1. Stratified Partitioning (70% Train, 15% Val, 15% Test)│
│ 2. Offline Training Augmentation (5x per sample)         │
│ 3. Save to dataset/prepared/{train,validation,test}/    │
└─────────────────────────────────────────────────────────┘
        │
        ▼ (02_Fit.ipynb)
┌─────────────────────────────────────────────────────────┐
│ 1. Load Batches via ImageDataGenerator (128x128 Rescaled)│
│ 2. Train Custom Multi-Class Sequential CNN              │
│ 3. Callbacks: ModelCheckpoint, EarlyStopping, ReduceLR  │
│ 4. Evaluate against pristine Test Split                  │
│ 5. Save best weights -> models/fruit_cnn_model.keras   │
└─────────────────────────────────────────────────────────┘
        │
        ▼ (03_Predict.ipynb)
┌─────────────────────────────────────────────────────────┐
│ 1. Load trained Keras model                             │
│ 2. Preprocess custom/real-world images from predict/     │
│ 3. Compute Softmax Probabilities & Predicted Class      │
│ 4. Render Dual-Panel Visualization (Image + Bar Chart)  │
└─────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```plaintext
Multilevel Classification CNN Model Project/
├── dataset/
│   ├── origin/                      # Raw, unaugmented image source
│   │   ├── apple/                   # Original Apple photographs
│   │   ├── banana/                  # Original Banana photographs
│   │   ├── mango/                   # Original Mango photographs
│   │   └── orange/                  # Original Orange photographs
│   └── prepared/                    # Partitioned & augmented working dataset
│       ├── train/                   # 228 images (38 original + 190 augmented)
│       ├── validation/              # 8 images (pure validation set)
│       └── test/                    # 9 images (untouched benchmark set)
├── models/
│   └── fruit_cnn_model.keras        # Best checkpointed trained model (Keras format)
├── predict/                         # Target images for inference testing
│   ├── apple_68.jpg
│   ├── Banana(lady_finger)_3.jpg
│   ├── mango_2.jpg
│   └── orange2.jpg
├── 01_Image_generator.ipynb         # Data splitting and offline augmentation
├── 02_Fit.ipynb                     # Model architecture, training, and metrics
├── 03_Predict.ipynb                 # Interactive visual inference and prediction
├── requirements.txt                 # Project library dependencies
└── README.md                        # Documentation and setup guide
```

---

## 📊 Dataset Specifications

### Original vs. Prepared Partitioning

| Fruit Class | Original Count | Train (Augmented) | Validation (Original) | Test (Original) |
| :--- | :---: | :---: | :---: | :---: |
| **Apple** | 20 | 84 (14 orig + 70 aug) | 3 | 3 |
| **Banana** | 13 | 54 (9 orig + 45 aug) | 2 | 2 |
| **Mango** | 8 | 30 (5 orig + 25 aug) | 1 | 2 |
| **Orange** | 14 | 60 (10 orig + 50 aug) | 2 | 2 |
| **Total** | **55** | **228** | **8** | **9** |

> [!IMPORTANT]
> **Zero Data Leakage Policy**: Splitting is executed **before** any data synthesis. No synthetic variants of validation or test samples ever enter the training partition.

---

## 🔄 Data Augmentation Pipeline

Configured in `01_Image_generator.ipynb` via Keras `ImageDataGenerator`:

| Augmentation Parameter | Value | Purpose / Physical Realism |
| :--- | :--- | :--- |
| **Normalization / Rescale** | `1.0 / 255.0` | Normalizes 8-bit RGB channels to `[0.0, 1.0]`. |
| **Rotation Range** | `15°` | Accommodates orientation variability on conveyor/surface. |
| **Zoom Range** | `±15%` | Simulates camera distance variations. |
| **Brightness Range** | `[0.85, 1.15]` | Accommodates ambient lighting and shadowing changes. |
| **Width & Height Shifts** | `±15%` | Handles off-center framing. |
| **Horizontal Flip** | `True` | Reflective symmetry invariant to fruit orientation. |
| **Fill Mode** | `nearest` | Avoids boundary edge artifacts upon shifting or rotating. |
| **Multiplier** | `5 variants` | Generates 5 distinct augmented PNGs per training image. |

---

## 🧠 CNN Architecture & Hyperparameters

The model is built using TensorFlow / Keras `Sequential` API with sequential feature hierarchy:

### Architectural Summary

```
Input (128, 128, 3)
  │
  ├── Conv2D (32 filters, 3x3, ReLU)
  ├── MaxPooling2D (2x2)
  │
  ├── Conv2D (64 filters, 3x3, ReLU)
  ├── MaxPooling2D (2x2)
  ├── Dropout (0.25)
  │
  ├── Flatten
  ├── Dense (128 units, ReLU)
  ├── Dropout (0.50)
  │
  └── Dense (4 units, Softmax) ──> Output Probabilities
```

### Layer Details

| Layer Name | Type | Output Shape | Parameters | Activation | Regularization |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `input_layer` | Input | `(None, 128, 128, 3)` | 0 | - | - |
| `conv2d_1` | Conv2D | `(None, 126, 126, 32)` | 896 | ReLU | - |
| `max_pool_1` | MaxPooling2D | `(None, 63, 63, 32)` | 0 | - | - |
| `conv2d_2` | Conv2D | `(None, 61, 61, 64)` | 18,496 | ReLU | - |
| `max_pool_2` | MaxPooling2D | `(None, 30, 30, 64)` | 0 | - | - |
| `dropout_1` | Dropout | `(None, 30, 30, 64)` | 0 | - | Rate = 0.25 |
| `flatten` | Flatten | `(None, 57600)` | 0 | - | - |
| `dense_1` | Dense | `(None, 128)` | 7,372,928 | ReLU | - |
| `dropout_2` | Dropout | `(None, 128)` | 0 | - | Rate = 0.50 |
| `dense_output` | Dense | `(None, 4)` | 516 | Softmax | - |

- **Total Parameters**: ~7,392,836
- **Trainable Parameters**: 7,392,836
- **Non-trainable Parameters**: 0

### Hyperparameters & Callbacks

- **Optimizer**: Adam (`learning_rate=0.001` initial)
- **Loss Function**: Sparse Categorical Crossentropy (`sparse_categorical_crossentropy`)
- **Batch Size**: 32
- **Target Resolution**: 128 × 128 × 3
- **Max Epochs**: 20
- **Dynamic Training Callbacks**:
  - `ModelCheckpoint`: Monitors `val_accuracy`, automatically saving the checkpoint with highest validation accuracy to `models/fruit_cnn_model.keras`.
  - `EarlyStopping`: Monitors `val_loss`, `patience=5`, with `restore_best_weights=True`.
  - `ReduceLROnPlateau`: Monitors `val_loss`, reducing learning rate by 50% (`factor=0.5`) after 2 stagnant epochs (`patience=2`, `min_lr=1e-6`).

---

## 📈 Model Performance & Evaluation

The final model weights (restored from optimal checkpoint at Epoch 6) were evaluated against the unseen **Test Partition** (9 independent images):

### Quantitative Results

- **Test Loss**: `0.3440`
- **Test Overall Accuracy**: `77.78%`

### Classification Report

```plaintext
              precision    recall  f1-score   support

       apple     1.0000    1.0000    1.0000         3
      banana     0.5000    0.5000    0.5000         2
       mango     0.5000    0.5000    0.5000         2
      orange     1.0000    1.0000    1.0000         2

    accuracy                         0.7778         9
   macro avg     0.7500    0.7500    0.7500         9
weighted avg     0.7778    0.7778    0.7778         9
```

### Analysis
- **Apples & Oranges**: 100% precision, recall, and F1-score due to distinct color profiles and round spatial shapes.
- **Bananas & Mangoes**: Subtle similarities in color tint and orientation across smaller sample profiles represent the primary margin for future fine-tuning.

---

## ⚙️ Environment Setup & Installation

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed.
- Recommended: 8 GB+ RAM and a modern multi-core CPU (or GPU).

### Step 1: Clone or Navigate to the Project Directory

```bash
cd "c:/Users/fhv/Downloads/Multilevel Classification CNN Model Project"
```

### Step 2: Create a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(If script execution is restricted in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first)*

**On Linux / macOS (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies

Upgrade `pip` and install all necessary dependencies using the included `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Contents of `requirements.txt`:
```plaintext
tensorflow>=2.15.0
scikit-learn>=1.3.0
numpy>=1.24.0
matplotlib>=3.7.0
pillow>=10.0.0
tqdm>=4.65.0
jupyterlab>=4.0.0
notebook>=7.0.0
```

---

## 🚀 Execution Guide

Run the pipeline in sequential order via Jupyter:

```bash
jupyter notebook
```
*(or use VS Code / Cursor Jupyter extension)*

### 1. Run Data Preparation (`01_Image_generator.ipynb`)
- Reads raw files from `dataset/origin/{apple,banana,mango,orange}`.
- Executes stratified splitting into `train`, `validation`, and `test`.
- Applies offline augmentation to `dataset/prepared/train/`.
- Verifies image counts across all folders.

### 2. Train the CNN Model (`02_Fit.ipynb`)
- Loads prepared directory data batches via `ImageDataGenerator(rescale=1./255)`.
- Compiles and trains the custom multi-class CNN architecture.
- Monitors convergence with Early Stopping and dynamic Learning Rate scaling.
- Evaluates against the untouched test set and plots loss/accuracy curves.
- Exports the best model to `models/fruit_cnn_model.keras`.

### 3. Run Inference (`03_Predict.ipynb`)
- Loads `models/fruit_cnn_model.keras`.
- Evaluates test photos placed inside `predict/`.
- Displays prediction output with confidence percentages and class probability bar charts.

---

## 🔮 Prediction & Inference Guide

### Using the Prediction Notebook
1. Drop any fruit image (format: `.jpg`, `.jpeg`, `.png`, `.webp`, or `.bmp`) into the `predict/` directory.
2. Open `03_Predict.ipynb` and select your target image:
   ```python
   IMAGE_PATH = PREDICT_DIR / "apple_68.jpg"
   result = predict_image(IMAGE_PATH)
   ```
3. The function visualizes the image alongside a probability distribution bar chart.

### Standalone Python Script Inference
You can also run inference programmatically without opening Jupyter:

```python
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf

# Configurations
MODEL_PATH = Path("models/fruit_cnn_model.keras")
CLASS_NAMES = ["apple", "banana", "mango", "orange"]
IMAGE_SIZE = (128, 128)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_single_image(image_path: str):
    # Load and normalize
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB").resize(IMAGE_SIZE)
    
    batch = np.expand_dims(np.asarray(img_rgb) / 255.0, axis=0)
    
    # Predict
    probabilities = model.predict(batch, verbose=0)[0]
    top_index = np.argmax(probabilities)
    
    predicted_class = CLASS_NAMES[top_index]
    confidence = probabilities[top_index]
    
    print(f"\nResult: {predicted_class.upper()} ({confidence:.2%})")
    for fruit, prob in zip(CLASS_NAMES, probabilities):
        print(f" - {fruit.capitalize():<8}: {prob:.2%}")
        
    return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    predict_single_image("predict/apple_68.jpg")
```

### Sample Prediction Outputs

```plaintext
Input: predict/apple_68.jpg
Prediction: Apple (97.84%)
 - Apple   : 97.84%
 - Banana  : 1.97%
 - Mango   : 0.15%
 - Orange  : 0.04%

Input: predict/Banana(lady_finger)_3.jpg
Prediction: Banana (97.58%)
 - Apple   : 0.12%
 - Banana  : 97.58%
 - Mango   : 1.46%
 - Orange  : 0.85%

Input: predict/orange2.jpg
Prediction: Orange (87.20%)
 - Apple   : 1.47%
 - Banana  : 5.06%
 - Mango   : 6.26%
 - Orange  : 87.20%
```

---

## 💡 Technical Highlights & Best Practices

- **Strict Random State Management**: A fixed seed (`42`) is applied across NumPy, Scikit-learn, and TensorFlow to guarantee repeatable splits and weight initializations.
- **Modern Keras Format**: Models are saved in the modern, cross-platform Keras v3 archive format (`.keras`) instead of legacy H5 or SavedModel directory formats.
- **Sparse Categorical Crossentropy**: Utilizes memory-efficient integer indexing for class labels without allocating high-cardinality one-hot vectors in batch iterators.
- **Double Regularization**: Employs spatial Dropout (`0.25`) after convolutional pooling and dense Dropout (`0.50`) prior to classification to curb overfitting on small datasets.

---

## 🛠️ Troubleshooting & FAQs

#### Q1: TensorFlow warning regarding GPU on Windows?
> `WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11.`

**Solution**: Native Windows support for GPU was deprecated in TensorFlow 2.11+. For this project, dataset sizes run efficiently on CPU. For GPU acceleration on Windows, utilize **WSL2 (Windows Subsystem for Linux)** or install `tensorflow-directml-plugin`.

#### Q2: `FileNotFoundError: Model not found`?
**Solution**: Ensure you have executed `02_Fit.ipynb` completely to create the `models/fruit_cnn_model.keras` checkpoint file before running `03_Predict.ipynb`.

#### Q3: WebP or PNG transparency issues?
**Solution**: The prediction pipeline converts all incoming images via `.convert("RGB")` in Pillow, safely discarding alpha channels and preventing 4-channel tensor shape mismatches.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
