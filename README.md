# 🫀 Automated Cardio-Respiratory Sound Analysis for Disease Screening

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Deep_Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Librosa-00C7B7?style=for-the-badge&logo=python&logoColor=white"/>
</p>

> An AI-powered cardio-respiratory sound classification system using MFCC feature extraction, Deep Neural Networks, and hybrid ML models for automated disease screening.

---

## 📌 Project Overview

This project builds an intelligent system that **automatically analyzes heart and lung sounds** to detect cardio-respiratory diseases — specifically **Aortic Stenosis** and **COPD**. By leveraging **MFCC feature extraction** and a suite of ML/DL models, this system assists in early disease screening with a best accuracy of **95.69%**.

---

## 🎯 Key Features

- 🎙️ **Audio Upload** — Upload `.wav` sound recordings for real-time prediction
- 🔬 **MFCC Feature Extraction** — Extracts meaningful features from raw audio signals
- 🤖 **Multiple ML Models** — DNN, FFBP-SVM, GLVQ, Perceptron trained & compared
- 📊 **Model Comparison Dashboard** — Compare all models side by side
- 🌐 **Streamlit Web App** — Interactive interface for disease screening

---

## 🏆 Model Results

| # | Model | Accuracy | Precision | Recall | F1 Score |
|---|-------|----------|-----------|--------|----------|
| 🥇 | **FFBP-SVM** | **95.69%** | **96.06%** | **95.17%** | **95.54%** |
| 🥈 | DNN Classifier | 82.76% | 82.90% | 83.70% | 82.68% |
| 🥉 | GLVQ | 57.76% | 57.50% | 57.68% | 57.38% |
| 4 | Perceptron | 44.83% | 71.68% | 52.24% | 34.53% |

> 🏆 **FFBP-SVM achieved 95.69% accuracy** — the best performing model for cardio-respiratory disease classification!

---

## 🎯 Disease Classes

| Class | Disease |
|-------|---------|
| 0 | Aortic Stenosis |
| 1 | COPD (Chronic Obstructive Pulmonary Disease) |

---

## 📊 Dataset

- **Classes:** 2 (Aortic Stenosis, COPD)
- **Total Samples:** 580 audio recordings
- **Train Set:** 464 samples
- **Test Set:** 116 samples
- **Format:** `.wav` audio files

---

## 🏗️ System Architecture

```
Audio Input (.wav)
     ↓
Preprocessing & Noise Reduction
     ↓
MFCC Feature Extraction
     ↓
Model Training (DNN / FFBP-SVM / GLVQ / Perceptron)
     ↓
Model Comparison & Evaluation
     ↓
Best Model Prediction → Disease Classification
     ↓
Streamlit Web App Output
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Audio Processing | Librosa, SciPy |
| Feature Extraction | MFCC (Mel-Frequency Cepstral Coefficients) |
| Models | FFBP-SVM, DNN, GLVQ, Perceptron |
| ML Framework | Scikit-learn |
| Web App | Streamlit |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## 📁 Project Structure

```
Automated-Cardio-Respiratory-Sound-Analysis/
│
├── app.py              # Streamlit web application
├── req.txt             # Required dependencies
├── background.jpg      # App background image
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/bharathkumararige/Automated-Cardio-Respiratory-Sound-Analysis-for-Disease-Screening.git
cd Automated-Cardio-Respiratory-Sound-Analysis-for-Disease-Screening
```

### 2. Install Dependencies
```bash
pip install -r req.txt
```

### 3. Run the Web App
```bash
streamlit run app.py
```

### 4. Open in Browser
```
http://localhost:8501
```

---

## 🧠 How to Use the App

1. 📁 **Upload Dataset** — Paste dataset folder path
2. 🎵 **Feature Extraction** — MFCC features extracted automatically
3. ✂️ **Train/Test Split** — Data split (80/20)
4. 🤖 **Train Models** — All 4 models trained simultaneously
5. 📊 **Compare Models** — View accuracy, precision, recall, F1 side by side
6. 🔮 **Predict** — Upload a `.wav` file for real-time disease prediction

---

## 🏥 Clinical Impact

Early detection of cardio-respiratory diseases through automated sound analysis can:
- ✅ Reduce diagnostic time significantly
- ✅ Enable screening in remote/low-resource areas
- ✅ Assist medical professionals with AI-powered second opinions
- ✅ Lower healthcare costs through early intervention

---

## 👨‍💻 Author

**Arige Bharath Kumar**
- 🎓 B.Tech CSE (Data Science) — Graduating July 2026
- 📧 [arigebharathkumar@gmail.com](mailto:arigebharathkumar@gmail.com)
- 🔗 [LinkedIn](https://linkedin.com/in/arigebharath)
- 🐙 [GitHub](https://github.com/bharathkumararige)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">⭐ <b>If you found this project helpful, please give it a star!</b> ⭐</p>
