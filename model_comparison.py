import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import joblib
import os

# Load your CSV
df = pd.read_csv("location_10_realtime_model_data.csv")

# Define features and targets
features = ["pH", "Tur", "Cond"]
targets = ["pH_next", "Tur_next", "Cond_next"]

results = {}
xgb_models = {}  # Dictionary to store XGBoost models


for target in targets:
    print(f"\n🔎 Evaluating for Target: {target}")
    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # LightGBM
    lgb_model = lgb.LGBMRegressor(random_state=42)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_mse = mean_squared_error(y_test, lgb_pred)

    # XGBoost
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_models[target] = xgb_model  # Save model

    # SARIMA (Univariate)
    sarima_order = (1, 1, 1)
    try:
        sarima_model = SARIMAX(
            df[target],
            order=sarima_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarima_results = sarima_model.fit(disp=False)
        sarima_pred = sarima_results.forecast(steps=len(y_test))
        sarima_mse = mean_squared_error(y_test.reset_index(drop=True), sarima_pred)
    except Exception as e:
        print(f"❌ SARIMA failed: {e}")
        sarima_mse = np.nan

    # Store results
    results[target] = {
        "LightGBM_MSE": round(lgb_mse, 5),
        "XGBoost_MSE": round(xgb_mse, 5),
        "SARIMA_MSE": round(sarima_mse, 5),
    }

# Display results as table
print("\n📊 Model Comparison Table (MSE):")
results_df = pd.DataFrame(results).T
print(results_df)

# Save best models (based on MSE comparison)
os.makedirs("models", exist_ok=True)
joblib.dump(xgb_models["pH_next"], "models/xgboost_ph.pkl")
joblib.dump(xgb_models["Tur_next"], "models/xgboost_tur.pkl")
joblib.dump(xgb_models["Cond_next"], "models/xgboost_cond.pkl")
