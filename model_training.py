"""
Model training module for water quality prediction
Handles Random Forest training, evaluation, and model persistence
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)
import joblib
import json
import config
import data_preprocessing as dp


def train_random_forest(X_train, y_train, use_grid_search=False):
    """
    Train Random Forest model

    Args:
        X_train (np.array): Training features
        y_train (np.array): Training target
        use_grid_search (bool): Whether to use grid search for hyperparameter tuning

    Returns:
        RandomForestClassifier: Trained model
    """
    if use_grid_search:
        print("Performing grid search for hyperparameter tuning...")
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        rf = RandomForestClassifier(random_state=config.RANDOM_STATE)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        model = grid_search.best_estimator_
    else:
        print("Training Random Forest with default parameters...")
        model = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
        model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, X_train=None, y_train=None):
    """
    Evaluate the trained model

    Args:
        model: Trained model
        X_test (np.array): Test features
        y_test (np.array): Test target
        X_train (np.array, optional): Training features for additional evaluation
        y_train (np.array, optional): Training target for additional evaluation

    Returns:
        dict: Evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation if training data provided
    cv_scores = None
    if X_train is not None and y_train is not None:
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=config.CROSS_VALIDATION_FOLDS
        )
        print(f"\nCross-validation scores: {cv_scores}")
        print(
            f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

    # Feature importance
    feature_importance = pd.DataFrame(
        {"feature": config.ALL_FEATURES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    # Prepare results dictionary
    results = {
        "test_accuracy": accuracy,
        "test_roc_auc": roc_auc,
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": feature_importance.to_dict("records"),
        "cross_validation_scores": (
            cv_scores.tolist() if cv_scores is not None else None
        ),
    }

    return results


def save_model(model, filepath=None):
    """
    Save the trained model

    Args:
        model: Trained model to save
        filepath (str, optional): Path to save the model
    """
    if filepath is None:
        filepath = config.MODEL_PATH

    try:
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model(filepath=None):
    """
    Load a trained model

    Args:
        filepath (str, optional): Path to load the model from

    Returns:
        Loaded model or None if error
    """
    if filepath is None:
        filepath = config.MODEL_PATH

    try:
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    except FileNotFoundError:
        print(f"Model file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def save_results(results, filepath=None):
    """
    Save evaluation results to JSON file

    Args:
        results (dict): Results dictionary
        filepath (str, optional): Path to save results
    """
    if filepath is None:
        filepath = config.RESULTS_PATH

    try:
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    """
    Main training pipeline
    """
    print("Starting water quality prediction model training...")

    # Load and preprocess data
    df = dp.load_data(config.DATA_PATH)
    if df is None:
        return

    # Clean data
    df_clean = dp.clean_data(df)
    if df_clean is None:
        return

    # Prepare features and target
    X, y = dp.prepare_features_and_target(df_clean)
    if X is None or y is None:
        return

    # Split data
    X_train, X_test, y_train, y_test = dp.split_data(X, y)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = dp.scale_features(X_train, X_test)

    # Train model
    model = train_random_forest(X_train_scaled, y_train, use_grid_search=False)

    # Evaluate model
    results = evaluate_model(model, X_test_scaled, y_test, X_train_scaled, y_train)

    # Save model and results
    save_model(model)
    save_results(results)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
