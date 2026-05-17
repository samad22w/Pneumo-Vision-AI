# 🫁 Pneumo-Vision AI
### AI-Powered Pneumonia Detection Using Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Accuracy](https://img.shields.io/badge/Accuracy-95.6%25-green)

---

## 📌 Project Overview
Pneumo-Vision AI is an AI-powered web application that detects 
pneumonia from chest X-ray images using Deep Learning. It provides 
instant diagnosis, visual heatmaps, and automated clinical PDF reports.

---

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | 95.6% |
| Validation Loss | 0.1366 |
| Test Accuracy | 90.87% |
| Sensitivity | 94.6% |
| Specificity | 84.6% |
| F1-Score (Pneumonia) | 0.93 |

---

## ✨ Features
- ⚡ Instant AI diagnosis from chest X-rays
- 🔥 Grad-CAM heatmaps — explainable AI
- 📄 Auto-generated PDF clinical reports
- 🩺 Doctor approval & notes system
- 📊 3-page Streamlit dashboard

---

## 🛠️ Technology Stack
- **Deep Learning:** TensorFlow / Keras
- **Model:** DenseNet121 (ImageNet pretrained)
- **Explainable AI:** Grad-CAM
- **Frontend:** Streamlit
- **Image Processing:** OpenCV, PIL
- **Report Generation:** FPDF2
- **Dataset:** Kaggle Chest X-Ray (5,856 images)

---

## 🚀 How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Dashboard
```bash
streamlit run app.py
```

---

## 📁 Project Structure