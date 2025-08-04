
# Lake Water Quality Prediction System
**End-to-End ML + IoT Project with ESP32 Deployment and Cloud Inference**

---

## 🚩 Project Overview

This project predicts three crucial lake water quality parameters — **pH, Turbidity, and Conductivity** — using machine learning models trained on large-scale lake data, and deploys the predictive models both on microcontrollers (ESP32) and cloud (AWS Lambda via Docker).

The workflow covers:
- Large-scale data collection, cleaning, and chunked processing
- Model training and evaluation (XGBoost, LightGBM, SARIMA, and LSTM)
- LSTM model training for multi-location time series forecasting using PyTorch and GPU acceleration
- Containerized FastAPI inference API with the LSTM model deployed on AWS Lambda using Docker
- Arduino/ESP32 firmware for field deployment with sensor integration and on-device inference

---

## 📊 Data & Problem Statement

- **Dataset:**  
  - ~80,000 lake locations  
  - Records from 2000 to 2023  
  - Each record includes pH, Turbidity (NTU), Conductivity (TDS, µS/cm)

- **Prototyping:**  
  - Initially focused on a single location with cleaned time series data  
  - Then extended to global multi-location data modeled with an LSTM

- **Cleaning:**  
  - Removed missing/outlier values  
  - Ensured physically plausible ranges (pH between 0 and 14, non-negative conductivity and turbidity)

- **Before and After Cleaning**

| Before Cleaning            | After Cleaning           |
|---------------------------|-------------------------|
| ![Before](./Small_Model/Figure_1.png) | ![After](./Small_Model/Figure_2.png) |

---

## 🧠 Model Training & Evaluation

### 1. Classical Models (Single Location)

| Algorithm | Type                  | Loss Function          | Outcome                  |
|-----------|-----------------------|-----------------------|--------------------------|
| XGBoost   | Gradient Boosting Tree| Mean Squared Error (MSE) | Selected best on single-location data |
| LightGBM  | Gradient Boosting Tree| Mean Squared Error (MSE) | Close runner-up           |
| SARIMA    | Time Series           | Likelihood/MSE         | Underperformed            |

---

### 2. LSTM Model (Multi-Location Global Forecasting)

- Implemented a **multi-step, multi-feature LSTM** in PyTorch.
- Trained on chunked datasets covering all locations, leveraging GPU acceleration.
- Trained to forecast next 12 months’ pH, Turbidity, and Conductivity given previous 12 months.
- Used advanced training strategies:
  - Mixed precision training (`torch.cuda.amp`)
  - Gradient scaling
  - Adaptive learning rate scheduler
  - Early stopping with patience
- Achieved improved generalization on large-scale data.
- Used ~5% of total data (~767,585 sequences) per epoch to fit memory constraints, from a total of over 15 million sequences.

---

## ☁️ Cloud Deployment: AWS Lambda with Docker

- Containerized the FastAPI inference server hosting the LSTM model.
- Used AWS Elastic Container Registry (ECR) for storing Docker images.
- Deployed container on AWS Lambda as a serverless function with:
  - Custom timeout and memory settings for model loading and inference.
  - Lambda API Gateway configured to route `/waterquality-lambda` POST requests.
- Supports scalable, on-demand inference accessible via HTTPS API.

---

## 🖥️ Local Testing

- Run the Docker container locally with:

  ```bash
  docker run -p 8000:8080 waterquality-api-lambda
````

* Test inference with:

  ```python
  import requests
  url = "http://localhost:8000/waterquality-lambda"
  data = {"input_sequence": [[7.5, 0.03, 5.0], ..., [7.51, 0.026, 5.00]]}
  response = requests.post(url, json=data)
  print(response.json())
  ```

---

## 🕹️ Microcontroller Deployment

* Exported XGBoost models to C header files using [m2cgen](https://github.com/BayesWitnesses/m2cgen).
* Developed firmware for ESP32 with:

  * Sensor interfacing (pH, Turbidity, Conductivity)
  * OLED display and button menu UI
  * On-device ML inference using exported models

---

## 🧩 Project Structure

```
.
├── data/
│   └── location_10_realtime_model_data.csv
├── models/
│   ├── multistep_lstm_best.pt
│   ├── xgboost_pH_next.json
│   ├── model_xgb_ph.h
│   └── ...
├── Small_Model/
│   ├── main.py
│   ├── model_comparison.py
│   ├── Figure_1.png
│   ├── Figure_2.png
│   └── ...
├── LSTM_Model/
│   ├── train.py
│   ├── chunked_dataset.py
│   ├── multistep_lstm_best.pt
│   └── ...
├── arduino/
│   ├── water_quality_predictor.ino
│   ├── model_xgb_ph.h
│   └── ...
├── lambda_deployment/
│   ├── Dockerfile
│   ├── app.py
│   ├── requirements.txt
│   └── multistep_lstm_best.pt
└── README.md
```

---

## 🚀 How to Reproduce

1. **Data Preparation & Cleaning**

2. **Train classical models** (run `Small_Model/model_comparison.py`) for initial prototyping.

3. **Train global LSTM** using chunked datasets and GPU (see `LSTM_Model/train.py`).

4. **Containerize LSTM inference API** (`lambda_deployment/Dockerfile`).

5. **Deploy Docker container to AWS Lambda and configure API Gateway**.

6. **Develop ESP32 firmware** for on-device inference and user interaction.

---

## 📚 References

* [PyTorch LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
* [AWS Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
* [FastAPI Documentation](https://fastapi.tiangolo.com/)
* [m2cgen Model to Code](https://github.com/BayesWitnesses/m2cgen)
* [Adafruit SSD1306 OLED Library](https://github.com/adafruit/Adafruit_SSD1306)

---

## 👤 Author

**Sahas Eashan**
University of Moratuwa, Sri Lanka

---

## 🚀 Future Work

* Extend to multiple location LSTM online predictions via REST API
* Full web app frontend for live data input and trend visualization
* Expand on-device ML capabilities with LSTM or TinyML models on ESP32


---

Would you like me to upload this as `README.md` for you to download?
```
