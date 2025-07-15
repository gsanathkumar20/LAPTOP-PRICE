# 💻 Laptop Price Prediction Using Machine Learning

Welcome to the **Laptop Price Predictor** – a machine learning-powered web application built with **Streamlit** that estimates the price of a laptop based on user-specified features such as RAM, screen size, CPU type, GPU brand, storage, and more.

---

## 🔍 Project Overview

Many users find it hard to estimate the price of a laptop based on specifications alone. This project uses machine learning regression techniques to predict laptop prices accurately based on various hardware and brand features.

Key stages of this project include:

* Data cleaning and preprocessing
* Feature engineering and encoding
* Model training with advanced regression algorithms
* Deployment using **Streamlit**

---

## 🚀 Demo

Launch the app locally using:

```bash
streamlit run app.py
```

> **Note:** Ensure `model.pkl` and `laptop_price.csv` are in the same directory as `app.py`.

---

## 🗂️ Project Structure

```plaintext
📁 laptop-price-prediction/
├── app.py                     # Streamlit app
├── model.pkl                  # Trained GradientBoostingRegressor model
├── laptop_price.csv           # Input dataset used for training and UI dropdowns
├── *.py                       # Notebooks converted to Python scripts
└── README.md                  # You're here!
```

---

## 📊 Dataset

The dataset includes various specifications of laptops such as:

* **Company**
* **TypeName** (Notebook, Gaming, etc.)
* **Operating System**
* **RAM, Weight, Inches**
* **CPU Brand & Speed**
* **GPU Brand**
* **Display Features** (Touchscreen, IPS, Resolution)
* **Storage** (HDD, SSD, Hybrid, Flash)
* **Price (in Euros)** — Target variable

---

## 🧠 Models Used

The following regression models were experimented with:

* **Linear Regression**
* **Random Forest Regressor**
* ✅ **Gradient Boosting Regressor** (Best performance, selected for deployment)

Performance metrics considered:

* MAE, MSE, RMSE
* R² score
* K-Fold cross-validation

---

## ⚙️ Features in the Web App

* Dynamic dropdowns and sliders for user input
* Real-time price prediction
* Visual price statistics (min, max, avg, median)
* Clean UI with styling enhancements using HTML & CSS

---

## 📦 Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, here are the main dependencies:

```txt
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## 🧪 Link to open the App:

https://gsanathkumar20-laptop-price-app-msxfph.streamlit.app/

---

## 📈 Example Prediction

Enter specs like:

* 16GB RAM
* 15.6" screen
* Intel i7 CPU
* Nvidia GPU
* 512GB SSD

And get a predicted price in Euros instantly!

---

## 📚 Acknowledgements

* Dataset sourced from Kaggle.
* Developed as part of an end-to-end ML project on regression modeling and deployment.

---

## 📌 Future Enhancements

* Add confidence intervals to predictions
* Include currency converter
* Allow CSV upload for batch predictions
* Deploy on Streamlit Cloud or HuggingFace Spaces

---

## 🧑‍💻 Author

Made with 💡 by **Sanath Kumar Guthikonda**

> *Aspiring Data Scientist | Passionate about ML Deployment | Streamlit Enthusiast*
