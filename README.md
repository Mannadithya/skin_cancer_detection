# 🧠 Skin Cancer Detection using EfficientNetB0

## 📌 Overview

This project focuses on automated skin lesion classification using deep learning. A transfer learning approach with EfficientNetB0 is used to classify dermoscopic images into 9 different skin disease categories from the ISIC dataset.

## 🎯 Objective

To build a robust and scalable deep learning model capable of assisting in early detection and classification of skin conditions, including various types of skin cancer.

## 🚀 Key Features

* Multi-class classification (9 skin disease categories)
* Transfer learning using EfficientNetB0
* Handles class imbalance using weighted loss
* Advanced data augmentation techniques
* Mixed precision training for efficiency
* Evaluation using ROC-AUC, accuracy, and confusion matrix

## 🧠 Model Details

* Architecture: EfficientNetB0 (pretrained on ImageNet)
* Fine-tuning of deeper layers for domain adaptation
* Loss Function: CrossEntropyLoss with class weights
* Optimizer: AdamW with cosine learning rate scheduler

## 📊 Results

* Validation Accuracy: ~52%
* ROC-AUC Score: ~0.89
* Stable training with no overfitting observed

## 📈 Visualizations

* Training vs Validation Loss
* Accuracy curves
* ROC Curve
* Confusion Matrix

## 🛠️ Technologies Used

* Python
* PyTorch
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn

## 📂 Dataset

ISIC (International Skin Imaging Collaboration) dataset from Kaggle, containing labeled dermoscopic images across multiple skin conditions.

## ▶️ How to Run

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:

   ```bash
   python app.py
   ```

## 🔮 Future Improvements

* Grad-CAM for model interpretability
* Deployment using Streamlit/Flask
* Binary classification for cancer detection
* Hyperparameter tuning for improved accuracy

## 📌 Conclusion

This project demonstrates the effectiveness of transfer learning in medical image classification and highlights the importance of evaluation metrics like ROC-AUC in healthcare applications.
