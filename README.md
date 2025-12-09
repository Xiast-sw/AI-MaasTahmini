# Salary Prediction AI

A machine learning application that predicts salaries based on experience, position, education level, age, and city using Polynomial Regression.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Project Overview

Salary Prediction AI is a desktop application that estimates salaries for software developers based on various factors. The model uses Polynomial Regression to capture non-linear relationships between features.

---

## 📊 Features Used

| Feature | Description | Values |
|---------|-------------|--------|
| **Experience** | Years of work experience | 0-20 years |
| **Position** | Job title | Junior, Mid, Senior Developer |
| **Education** | Education level | Lise, Ön Lisans, Lisans |
| **Age** | Employee age | 20-60 years |
| **City** | Work location | İstanbul, Ankara, İzmir |

---

## 🏗️ Project Structure

    SalaryPrediction-AI/
    ├── notebooks/
    │   └── SalaryPrediction.ipynb
    ├── src/
    │   ├── preprocessing.py
    │   ├── model.py
    │   ├── train.py
    │   └── app.py
    ├── results/
    │   ├── actual_vs_predicted.png
    │   └── experience_vs_salary.png
    ├── .gitignore
    ├── requirements.txt
    └── README.md

---

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| **Algorithm** | Polynomial Regression |
| **Polynomial Degree** | 2 |
| **Train/Test Split** | 80% / 20% |
| **Encoding** | One-Hot Encoding |

---

## 🚀 Getting Started

### 1. Clone the Repository

    git clone https://github.com/Xiast-sw/SalaryPrediction-AI.git
    cd SalaryPrediction-AI

### 2. Install Dependencies

    pip install -r requirements.txt

### 3. Run Training

    python src/train.py

### 4. Run Desktop App

    python src/app.py

### 5. Or Use Jupyter Notebook

    jupyter notebook notebooks/SalaryPrediction.ipynb

---

## 🖥️ Desktop Application

The GUI allows users to input:
- Years of experience
- Job position
- Education level
- Age
- City

And returns the **predicted salary** in TL.

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.x |
| **ML Library** | Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib |
| **GUI** | Tkinter |

---

## 📁 File Descriptions

| File | Description |
|------|-------------|
| `src/preprocessing.py` | Data generation and preprocessing |
| `src/model.py` | Model building and evaluation |
| `src/train.py` | Training script with visualization |
| `src/app.py` | Tkinter desktop application |

---

## 👤 Author

**Adil Buğra Aytar**

[![GitHub](https://img.shields.io/badge/GitHub-Xiast--sw-black?logo=github)](https://github.com/Xiast-sw)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Adil%20Buğra%20Aytar-blue?logo=linkedin)](https://linkedin.com/in/adil-bugra-aytar-47a555224)

[![Email](https://img.shields.io/badge/Email-a.bugraaytar@gmail.com-red?logo=gmail)](mailto:a.bugraaytar@gmail.com)

---

## 📝 License

This project is licensed under the MIT License.

