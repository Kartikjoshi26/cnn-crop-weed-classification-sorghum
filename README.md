# Sorghum Crop vs Weed Classification using CNN

A deep learning–based image classification project to identify Sorghum crops and weed types using a custom Convolutional Neural Network (CNN).

## Project Overview

Weed detection is a crucial task in precision agriculture. This project builds a multi-class classification model that distinguishes between:

- 🌾 Sorghum (Crop)
- 🌿 Grass Weeds
- 🍀 Broadleaf Weeds

The model is trained on an image dataset of sorghum fields and aims to assist in automated weed identification systems.

## Model Details

- **Framework:** TensorFlow / Keras
- **Architecture:** Custom CNN (5 Convolutional Layers)
- **Input Shape:** `(256, 256, 3)`
- **Output Classes:** 3
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Epochs:** 40
- **Batch Size:** 32

## Data Pipeline

### Dataset Loading

- Used `image_dataset_from_directory`
- Automatic label inference from folder structure

### Performance Optimization

- Caching
- Prefetching
- Shuffling

## Data Augmentation

Applied to improve generalization:

- Random horizontal flip
- Random rotation (±10%)

## Dataset Structure

```
SorghumWeedDataset_Classification/
│
├── Train/
│   ├── Sorghum/
│   ├── Grass/
│   └── BroadLeafWeed/
│
├── Validate/
│   ├── Sorghum/
│   ├── Grass/
│   └── BroadLeafWeed/
│
└── Test/
    ├── Sorghum/
    ├── Grass/
    └── BroadLeafWeed/
```

## Training Strategy

- **EarlyStopping** used to prevent overfitting
  - Monitors: `val_loss`
  - Patience: 5 epochs

## Evaluation Metrics

Model evaluated on test dataset using:

- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1 Score
- ✅ Confusion Matrix
- ✅ Classification Report

## Results

- Test Accuracy: 95.45%
- F1 Score: 96.55
- Confusion Matrix :
<img width="549" height="484" alt="image" src="https://github.com/user-attachments/assets/ae51418a-eff7-4e9e-b442-59ce56e38d5c" />


## Visualizations

- Training vs Validation Accuracy
<img width="803" height="536" alt="image" src="https://github.com/user-attachments/assets/b7ed2841-7e10-48ff-858f-223c808e416f" />

- Training vs Validation Loss
<img width="792" height="543" alt="image" src="https://github.com/user-attachments/assets/b493b74f-f7c2-4a7d-965e-2f5db3ad56b8" />


## Model Saving

Model saved as:

```
model_5conv.keras
```

## How to Run

1. Clone the repo:

```bash
git clone https://github.com/your-username/sorghum-weed-classification.git
cd sorghum-weed-classification
```

2. Install dependencies:

```bash
pip install tensorflow matplotlib scikit-learn
```

3. Run the notebook:

```bash
jupyter notebook Crop_vs_Weed_CNN_Classification_model.ipynb
```

## Future Improvements

- Use Transfer Learning (ResNet / EfficientNet)
- Convert to real-time weed detection (YOLO)
- Deploy on edge devices (Jetson Nano / mobile)
- Improve dataset diversity for robustness

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
