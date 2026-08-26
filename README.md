# 🎯 Concentration & Focus Predictor

An end-to-end Machine Learning solution and interactive web application designed to analyze personal work habits, digital distractions, screen time, and sleep quality to predict an individual's overall concentration ability (rated from 1 to 5).

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## 📌 Project Description

In modern digital environments, constant notifications, screen usage, multitasking, and irregular sleep schedules significantly degrade deep work capacity. 

This project leverages survey data from students and working professionals to construct a predictive Random Forest Classifier that estimates personal concentration levels. Alongside predictions, the built-in **Streamlit Web Application** provides personalized productivity recommendations (such as phone placement strategies, notification silencing, and sleep hygiene) to help users regain focus.

---

## 🌟 Key Features

- **Predictive ML Model**: Uses 200 decision trees (`RandomForestClassifier`) to classify focus ability into 5 rating tiers.
- **Interactive Streamlit Web Dashboard**: Sleek, minimalistic form interface built for real-time predictions.
- **Actionable Productivity Feedback**: Instant tailored tips based on user-selected habits (e.g. phone checks, multitasking, notification frequency).
- **Feature Importance Analytics**: Visual breakdown highlighting top drivers influencing concentration.
- **Sanitized Encoding**: Clean handling of non-ASCII characters and dataset encodings.

---

## 📊 Dataset Parameters

The model evaluates **15 distinct behavioral indicators**:

| Category | Features Evaluated |
| :--- | :--- |
| **Profile & Schedule** | Age Group, Gender, Role, Daily Sleep Duration, Morning Refreshment |
| **Digital Distractions** | Daily Screen Time, Primary Distraction Type, Phone Check Frequency, Notification Interruptions |
| **Work Habits & Focus** | Focused Session Duration, Multitasking Frequency, Hourly Break Count, Time to Regain Focus, Peak Distraction Time |
| **Target Output** | Overall Concentration Ability (1 - 5 Rating Score) |

---

## 📁 Repository Structure

```
.
├── app.py                # Streamlit interactive web interface
├── train_model.py        # Model training, preprocessing & export script
├── Focus.ipynb           # Original Jupyter notebook with EDA & training
├── focus.csv             # Survey dataset (83 responses)
├── model.pkl             # Pre-trained Random Forest model & label encoders
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation & description
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/prathaam17/Concentration-Focus-Predictor.git
cd Concentration-Focus-Predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train / Retrain Model
```bash
python train_model.py
```

### 4. Run the Web Application
```bash
streamlit run app.py
```
Access the UI at `http://localhost:8501`.

---

## 👤 Author

Developed by **Pratham** ([@prathaam17](https://github.com/prathaam17))

## 📜 License

This project is licensed under the [MIT License](LICENSE).
