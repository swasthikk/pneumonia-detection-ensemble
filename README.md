# 🩺 Pneumonia Detection using Deep Learning

A deep learning-based system to detect pneumonia from chest X-ray images using transfer learning and ensemble techniques.

---

## 🚀 Overview

This project focuses on building an accurate and reliable pneumonia detection model using multiple CNN architectures and combining them using an ensemble approach.

We experimented with:

* EfficientNetB0
* MobileNetV2 
* Ensemble Model

The goal was to compare models and build a robust system suitable for real-world medical applications.

---

## 📂 Dataset

* Dataset: Chest X-Ray Pneumonia Dataset
* Source: Kaggle
* Total Images: **5216**
* Classes:

  * Normal
  * Pneumonia

---

## ⚙️ Methodology

### 🔹 Data Processing

* Image resizing (224x224)
* Normalization
* Class imbalance handled using class weights

### 🔹 Model Training

* Transfer learning with ImageNet weights
* Fine-tuning of top layers
* Stratified K-Fold Cross Validation (3 folds)

### 🔹 Evaluation Metrics

* Accuracy
* AUC (Area Under Curve)
* Precision, Recall, F1-score
* Confusion Matrix

### 🔹 Ensemble Technique

* Stacking (Meta-Learner: Logistic Regression)
* Combines predictions from MobileNet and EfficientNet

---

## 📊 Results

| Model          | Accuracy   | AUC        | Remarks           |
| -------------- | ---------- | ---------- | ----------------- |
| EfficientNetB0 | 81.36%     | 0.828      | Moderate performance  |
| MobileNetV2    | 93.25%     | 0.9819     | Best single model |
| Ensemble Model | **96.22%** | **0.9872** | Best overall 🚀   |

---

## 📈 Key Insights

* MobileNet performed best individually due to better generalization on limited data.
* EfficientNet underperformed due to higher complexity and preprocessing sensitivity.
* Ensemble model improved accuracy and reduced false negatives.
* In medical AI, reducing missed cases (false negatives) is critical.

---

## 🧠 Conclusion

The ensemble model combining MobileNet and EfficientNet achieved the best performance, making it suitable for real-world pneumonia detection systems.

Even though EfficientNet alone performed poorly, it contributed useful features in the ensemble.

---

## 🔮 Future Work

* Apply Grad-CAM for model explainability
* Use lung segmentation for better focus
* Train on larger and more diverse datasets
* Deploy as a web application for real-time predictions

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* NumPy, Matplotlib, Seaborn

---

## 📌 How to Run

1. Clone the repository:

```bash
git clone https://github.com/swasthikk/pneumonia-detection-ensemble.git
cd pneumonia-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
jupyter notebook
```

---

## 📬 Contact

If you have any questions or suggestions, feel free to reach out.

---

⭐ If you found this project useful, consider giving it a star!
