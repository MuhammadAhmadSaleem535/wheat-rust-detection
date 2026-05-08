# 🌾 Wheat Rust Detection and Severity Estimation

## 📌 Overview
This project is a deep learning-based system designed to detect wheat rust diseases and estimate their severity from leaf images. It provides an end-to-end pipeline that not only identifies the type of rust but also assesses how severe the infection is and generates actionable recommendations for crop management.

## 🧠 Project Features
- Wheat rust disease classification:
  - Black Rust
  - Brown Rust
  - Yellow Rust
  - Healthy
- Severity estimation:
  - Moderate
  - Severe
- Decision support system for agricultural recommendations based on prediction results

## ⚙️ Methodology
- Transfer learning using ResNet18 convolutional neural network
- Image preprocessing: resizing and normalization
- Two separate models:
  - Disease classification model
  - Severity estimation model
- Rule-based recommendation system for treatment guidance

## 📊 Dataset
The model is trained on a wheat leaf disease dataset containing images of rust-infected and healthy wheat plants. The dataset is split into training, validation, and test sets.

## 🚀 How it works
1. Input wheat leaf image
2. Image preprocessing
3. Rust disease classification
4. Severity estimation
5. Recommendation generation based on predictions

## 📁 Files Included
- `main.py` → Full inference pipeline

## 🎯 Goal
To assist farmers and agricultural experts in early detection of wheat rust diseases and provide guidance for effective crop management.

## 👨‍💻 Author
- Muhammad Ahmad Saleem
  
