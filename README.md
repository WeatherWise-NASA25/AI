# WeatherWise - Weather Forecasting ML Models

*WeatherWise* is a Python-based weather forecasting project that leverages machine learning techniques, including *MLP (Multi-Layer Perceptron)* and *LSTM (Long Short-Term Memory)* models, to predict weather conditions such as temperature, humidity, and precipitation for different cities. This project focuses on both tabular and sequential modeling for accurate predictions based on historical weather data.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Data Requirements](#data-requirements)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Machine Learning Models](#machine-learning-models)
7. [Prediction](#prediction)
8. [License](#license)

---

## Features

* Load and merge multiple CSV datasets containing historical weather data.
* Preprocess data and engineer features including:

  * Date features (year, month, day)
  * Seasons (Winter, Spring, Summer, Fall)
  * Cyclical features for month and day
* Train *MLP* and *LSTM* regression models for predicting:

  * Temperature (temperature_c)
  * Humidity (humidity_percent)
  * Precipitation (precipitation_mm)
* Generate city-specific sequences for LSTM training to avoid data leakage.
* Evaluate models with *Mean Absolute Error (MAE)* and *Root Mean Squared Error (RMSE)* metrics.
* Make predictions for any city and date using historical averages.
* Export predictions to CSV.

---

## Installation

1. Clone this repository:

   bash
   git clone <repository_url>
   cd WeatherWise
   

2. Install required packages:

   bash
   pip install -r requirements.txt
   

3. Ensure you have *Python 3.8+* installed.

*Required Libraries:*

* pandas, numpy, torch, scikit-learn, joblib, glob, os

---

## Data Requirements

* All historical weather CSV files should be placed in the same folder.
* CSVs should include at least the following columns:

  
  city, date, temperature_c, humidity_percent, precipitation_mm
  
* Files containing the word prediction will be ignored during training.

---

## Usage

1. Run the main script to train models and make example predictions:

   bash
   python weatherwise.py
   

   This will:

   * Load and preprocess data
   * Train MLP and LSTM models
   * Evaluate models
   * Generate a sample prediction for New York on 2025-07-20
   * Export predictions to predictions.csv

2. To make predictions for other cities or dates, use the predict_weather function:

   python
   from weatherwise import predict_weather
   results = predict_weather(city='London', date_str='2025-10-15', full_df=df, export_to_csv=True)
   print(results)
   

---
## Machine Learning Models

### 1. MLP (Multi-Layer Perceptron)

* Fully connected feed-forward neural network.
* Architecture:

  * Input layer → 128 → 64 → Output layer
  * ReLU activations
  * Dropout for regularization

### 2. LSTM (Long Short-Term Memory)

* Sequence model for capturing temporal dependencies.
* Architecture:

  * 2 LSTM layers with 64 hidden units
  * Dropout for regularization
  * Fully connected output layer

### Evaluation Metrics

* *Mean Absolute Error (MAE)*
* *Root Mean Squared Error (RMSE)*

---

## Prediction

* Predictions are based on historical averages for a specific city and month.
* Output is a dictionary with predicted values for temperature, humidity, and precipitation:

  python
  {
      "temperature_c": "25.32",
      "humidity_percent": "60.50",
      "precipitation_mm": "2.10"
  }
  
* Optionally, predictions can be exported to predictions.csv.

---

## License

This project is licensed under the MIT License.
Feel free to use, modify, and distribute!