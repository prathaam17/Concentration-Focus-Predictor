# 🎯 Concentration & Focus Predictor

A Machine Learning project and interactive Streamlit web application designed to analyze user habits, digital distractions, and sleep patterns to predict overall concentration ability (on a scale of 1 to 5) and provide actionable productivity tips.

---

## 🌟 Features

- **Machine Learning Model**: Built with `RandomForestClassifier` trained on survey dataset (`focus.csv`).
- **Interactive Web Interface**: Streamlit UI with clean form inputs, instant predictions, score badges, and personalized focus recommendations.
- **Factor Importance Analysis**: Visual bar chart detailing key drivers impacting focus (e.g. screen time, phone checks, notification interruptions).
- **Sanitized Encoding**: Clean handling of non-ASCII characters and dataset encodings.

---

## 📁 Repository Structure

```
.
├── app.py                # Streamlit web application interface
├── train_model.py        # Model training and artifact export script
├── Focus.ipynb           # Original research notebook & EDA
├── focus.csv             # Survey dataset
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation

Ensure you have Python 3.9+ installed. Clone the repository and install dependencies:

```bash
git clone https://github.com/prathaam17/Concentration-Focus-Predictor.git
cd Concentration-Focus-Predictor
pip install -r requirements.txt
```

### 2. Train the Model

Train the Random Forest model and generate serialized artifacts (`model.pkl`):

```bash
python train_model.py
```

### 3. Run the Web Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📊 Dataset & Model Overview

- **Dataset**: `focus.csv` (83 survey responses with 18 lifestyle & focus attributes).
- **Target**: `14.Rate your overall concentration ability` (1 to 5 rating).
- **Classifier**: Random Forest Classifier (200 decision trees).

---

## 🛠️ Built With

- **Python**
- **Streamlit**
- **Scikit-Learn**
- **Pandas & NumPy**
- **Matplotlib**
