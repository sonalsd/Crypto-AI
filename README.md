# Crypto AI Price Prediction System

## Project Overview
This project is to predicts cryptocurrency prices using a trained model. 
The project includes a Flask web application for user interaction and prediction.


## Features
- Crypto price prediction using trained ML model
- Data preprocessing using scaler
- Flask-based web interface
- Simple and interactive UI
- Ready for deployment

## Project Structure

```bash
crypto-Ai/
│
├── templates/
│   └── index.html
│
├── venv/
│
├── app.py
├── crypto_model.h5
├── scaler.gz
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository
git clone https://github.com/your-username/crypto-ai.git

cd crypto-ai

### 2. Create virtual environment
python -m venv venv

Activate it:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

## Run the Project
python app.py

Then open:
http://127.0.0.1:5000

## Requirements
Flask
numpy
pandas
scikit-learn
tensorflow
joblib

## Model Details
- Type: Machine Learning / Deep Learning Model
- Framework: TensorFlow / Keras
- Input: Scaled crypto market data
- Output: Predicted price

## Future Improvements
- Live crypto API integration
- Better ML models (LSTM/GRU)
- Cloud deployment (Render / AWS)
- Improved UI dashboard

## License
This project is for educational purposes only.
