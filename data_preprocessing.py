"""
Data preprocessing module for water quality prediction
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import config


def load_data(file_path):
    """
    Load water quality data from CSV file

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def clean_data(df):
    """
    Clean the dataset by handling missing values and outliers

    Args:
        df (pd.DataFrame): Raw dataframe

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if df is None:
        return None

    print(f"Original data shape: {df.shape}")

    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values per column:\n{missing_values}")

    # Fill missing values with median for numerical columns
    for column in config.ALL_FEATURES:
        if column in df.columns and df[column].isnull().sum() > 0:
            median_val = df[column].median()
            df[column].fillna(median_val, inplace=True)
            print(
                f"Filled {missing_values[column]} missing values in {column} with median: {median_val:.2f}"
            )

    # Remove extreme outliers using IQR method
    cleaned_df = df.copy()

    for column in config.ALL_FEATURES:
        if column in cleaned_df.columns:
            Q1 = cleaned_df[column].quantile(0.25)
            Q3 = cleaned_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_before = len(cleaned_df)
            cleaned_df = cleaned_df[
                (cleaned_df[column] >= lower_bound)
                & (cleaned_df[column] <= upper_bound)
            ]
            outliers_removed = outliers_before - len(cleaned_df)

            if outliers_removed > 0:
                print(f"Removed {outliers_removed} outliers from {column}")

    print(f"Final cleaned data shape: {cleaned_df.shape}")
    return cleaned_df


def prepare_features_and_target(df):
    """
    Prepare features and target variables for training

    Args:
        df (pd.DataFrame): Cleaned dataframe

    Returns:
        tuple: (X, y) where X is features and y is target
    """
    if df is None:
        return None, None

    # Select features and target
    X = df[config.ALL_FEATURES].copy()
    y = df[config.TARGET_COLUMN].copy()

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    return X, y


def scale_features(X_train, X_test, save_scaler=True):
    """
    Scale features using StandardScaler

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        save_scaler (bool): Whether to save the scaler

    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if save_scaler:
        joblib.dump(scaler, config.SCALER_PATH)
        print(f"Scaler saved to {config.SCALER_PATH}")

    return X_train_scaled, X_test_scaled, scaler


def split_data(X, y):
    """
    Split data into training and testing sets

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


def get_data_statistics(df):
    """
    Get basic statistics about the dataset

    Args:
        df (pd.DataFrame): Dataframe to analyze

    Returns:
        dict: Statistics dictionary
    """
    if df is None:
        return {}

    stats = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "basic_stats": df.describe().to_dict(),
        "target_distribution": (
            df[config.TARGET_COLUMN].value_counts().to_dict()
            if config.TARGET_COLUMN in df.columns
            else {}
        ),
    }

    return stats


def load_scaler():
    """
    Load the saved scaler

    Returns:
        StandardScaler: Loaded scaler object
    """
    try:
        scaler = joblib.load(config.SCALER_PATH)
        print(f"Scaler loaded from {config.SCALER_PATH}")
        return scaler
    except FileNotFoundError:
        print(f"Scaler file not found at {config.SCALER_PATH}")
        return None
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None
