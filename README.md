# Stock Market Forecasting Web Application

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Prophet%2C%20SARIMAX%2C%20Random%20Forest%2C%20LSTM-orange)


A web application for predicting stock prices using machine learning models. Built with Python, Flask, and Plotly for interactive visualizations.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)

---

## Overview
This project is a **Stock Market Forecasting Web Application** that predicts future stock prices using historical data. It supports multiple machine learning models, including **Prophet**, **SARIMAX**, **Random Forest**, and **LSTM**. The application provides an intuitive user interface for selecting stocks, choosing models, and visualizing predictions with interactive plots.

---

## Features
- **Multiple Forecasting Models**:
  - Prophet
  - SARIMAX
  - Random Forest
  - LSTM (Long Short-Term Memory)
- **Interactive Visualizations**: Plotly-based graphs for historical and predicted prices.
- **Error Analysis**: Visualize the error between actual and predicted prices.
- **Caching Mechanism**: Stores historical stock data locally to reduce API calls.
- **User-Friendly Interface**: Built with Bootstrap for a clean and responsive design.

---

## Technologies Used
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, Bootstrap
- **Machine Learning**: Prophet, SARIMAX, Random Forest, LSTM (TensorFlow/Keras)
- **Data Fetching**: Yahoo Finance API (`yfinance`)
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Error Metrics**: Scikit-learn (Mean Absolute Error)

---

## How It Works
1. **User Input**: Enter a stock ticker (e.g., AAPL, TSLA), select a model, and specify the number of days to predict.
2. **Data Fetching**: Historical stock data is fetched using the Yahoo Finance API and cached locally.
3. **Model Training**: The selected model is trained on the historical data.
4. **Prediction**: The model generates future stock price predictions with confidence intervals.
5. **Visualization**: Interactive plots are displayed on the web interface.
6. **Error Analysis**: The application calculates and visualizes the error between actual and predicted prices.

---

## Project Structure
```
📁 stock-forecasting-app
│-- app.py                 # Main Flask application
│-- models.py              # Machine learning models
│-- data_fetcher.py        # Fetch stock data from Yahoo Finance
│-- templates/             # HTML templates
│-- static/                # CSS, JS, and images
│-- requirements.txt       # Dependencies
│-- README.md              # Project documentation
```

## 🔧 Installation & Setup
### Prerequisites
- Python 3.8+
- Virtual Environment (recommended)

## 📊 Usage
1. Enter a stock ticker symbol (e.g., AAPL, TSLA) in the web interface.
2. Choose a prediction model.
3. Click 'Predict' to get the forecasted stock price.
4. View historical and predicted prices on the visualization chart.

## 🖼️ Screenshots

### 📌 Home Page
![Home Page](https://github.com/hamzafareed12/Stock_Market_Forecasting/blob/master/frontend.png)

### 📌 Prediction Results
![Prediction Results](https://github.com/hamzafareed12/Stock_Market_Forecasting/blob/master/prediction.png)

### 📌 Prediction Graph
![Prediction Results](https://github.com/hamzafareed12/Stock_Market_Forecasting/blob/master/prediction_graph.png)

### 📌 Error Graph
![Prediction Results](https://github.com/hamzafareed12/Stock_Market_Forecasting/blob/master/error_graph.png)

## 🎯 Future Enhancements
- Add more advanced deep learning models (e.g., LSTM, GRU)
- Improve UI with interactive charts (Plotly, D3.js)
- Implement authentication for personalized forecasts

  ## 📬 Contact
For any queries, feel free to reach out:
- **Email**: hamzafareed7@gmail.com
- **GitHub**: [https://github.com/hamzafareed12](https://github.com/hamzafareed12)
- **LinkedIn**: [muhammad-hamza-9b256b162/](https://linkedin.com/in/muhammad-hamza-9b256b162/)
