"""
Configuration file for Water Quality Prediction Model
Contains all constants, file paths, and optimal parameter values
"""

import os

# File paths
DATA_PATH = (
    r"C:\Users\Cyborg\Documents\GitHub\Water-Quality-MEasuring-ML\water_potability.csv"
)
MODEL_PATH = "trained_model.pkl"
SCALER_PATH = "scaler.pkl"
RESULTS_PATH = "model_results.json"

# Feature columns
ALL_FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

# Features that will be measured (available for prediction)
MEASURED_FEATURES = ["ph", "Solids", "Turbidity"]

# Features that will be constrained to optimal values
CONSTRAINED_FEATURES = [
    "Hardness",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
]

TARGET_COLUMN = "Potability"

# Optimal values for constrained features (based on WHO standards and typical good water quality)
OPTIMAL_VALUES = {
    "Hardness": 150.0,  # mg/L as CaCO3 (moderately hard water)
    "Chloramines": 2.0,  # mg/L (safe disinfection level)
    "Sulfate": 200.0,  # mg/L (well below WHO guideline of 500 mg/L)
    "Conductivity": 400.0,  # μS/cm (good conductivity for drinking water)
    "Organic_carbon": 2.0,  # mg/L (low organic carbon content)
    "Trihalomethanes": 50.0,  # μg/L (well below WHO guideline of 100 μg/L)
}

# Model parameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CROSS_VALIDATION_FOLDS = 5

# Validation ranges for measured features (for input validation)
FEATURE_RANGES = {"ph": (0.0, 14.0), "Solids": (0.0, 50000.0), "Turbidity": (0.0, 10.0)}
