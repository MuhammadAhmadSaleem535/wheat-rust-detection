# 🌾 Wheat Disease Detection and Severity Estimation

## 📌 Overview

This project is a deep learning-based system designed to detect wheat diseases from leaf images and estimate their severity. It provides an end-to-end pipeline that identifies the disease type, evaluates how severe the infection is, and generates useful recommendations for crop management.

---

## 🚀 Features

* 🌿 Multi-class wheat disease classification (**15 classes + healthy**)
* ⚠️ Severity estimation:

  * Moderate
  * Severe
* 📊 Confidence scores for predictions
* 🤖 AI-based treatment and recommendation system
* 💻 Simple and user-friendly interface

---

## 🧠 Methodology

* Transfer learning using convolutional neural networks (CNNs)
* Models used:

  * ResNet18
  * MobileNetV3
* Image preprocessing:

  * Resizing to 224 × 224
  * Normalization
  * Data augmentation (flip, rotation, color jitter)
* Two-model pipeline:

  * Disease classification model
  * Severity classification model

---

## ⚙️ How It Works

1. Input wheat leaf image
2. Image preprocessing
3. Disease classification
4. Severity prediction
5. Recommendation generation

---

## 📊 Results

* **Disease Classification Accuracy:** 91%
* **Severity Prediction Accuracy:** 87%

The model performs well on major diseases, with minor confusion between visually similar classes.

---

## ⚠️ Challenges

* Class imbalance in dataset
* Visually similar diseases
* Limited data for some classes

---

## 🔮 Future Improvements

* Increase dataset size and diversity
* Improve accuracy on similar disease classes
* Mobile application deployment
* Real-world field testing

---

## 📂 Project Structure

* `main.py` → Full inference pipeline
* `train.py` → Disease model training
* `severity_train.py` → Severity model training
* `models/` → Saved trained models

---

## 🎯 Goal

To assist farmers and agricultural experts in early detection of wheat diseases and provide practical guidance for effective crop management.

---

## 👨‍💻 Authors

* Muhammad Ahmad Saleem


---

## 📌 Note

This project is developed for educational purposes and demonstrates the application of deep learning in agriculture.

---
