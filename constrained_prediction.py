"""
Constrained prediction module for water quality assessment
Handles predictions using only measured features (pH, Solids, Turbidity)
with optimal values for other parameters
"""

import pandas as pd
import numpy as np
import joblib
import config
import model_training as mt
import data_preprocessing as dp


class ConstrainedPredictor:
    """
    Predictor class that uses only measured features and optimal values for constrained features
    """

    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialize the predictor

        Args:
            model_path (str, optional): Path to trained model
            scaler_path (str, optional): Path to scaler
        """
        self.model = None
        self.scaler = None
        self.is_ready = False

        self.load_model(model_path)
        self.load_scaler(scaler_path)

    def load_model(self, model_path=None):
        """
        Load the trained model

        Args:
            model_path (str, optional): Path to model file
        """
        if model_path is None:
            model_path = config.MODEL_PATH

        self.model = mt.load_model(model_path)
        if self.model is not None:
            print("Model loaded successfully for constrained prediction")
            self._check_readiness()

    def load_scaler(self, scaler_path=None):
        """
        Load the scaler

        Args:
            scaler_path (str, optional): Path to scaler file
        """
        if scaler_path is None:
            scaler_path = config.SCALER_PATH

        self.scaler = dp.load_scaler()
        if self.scaler is not None:
            print("Scaler loaded successfully for constrained prediction")
            self._check_readiness()

    def _check_readiness(self):
        """
        Check if predictor is ready for predictions
        """
        self.is_ready = self.model is not None and self.scaler is not None
        if self.is_ready:
            print("Constrained predictor is ready for predictions")

    def validate_input(self, ph, solids, turbidity):
        """
        Validate input parameters

        Args:
            ph (float): pH value
            solids (float): Total dissolved solids
            turbidity (float): Turbidity value

        Returns:
            tuple: (is_valid, error_message)
        """
        errors = []

        # Check pH range
        if not (config.FEATURE_RANGES["ph"][0] <= ph <= config.FEATURE_RANGES["ph"][1]):
            errors.append(
                f"pH must be between {config.FEATURE_RANGES['ph'][0]} and {config.FEATURE_RANGES['ph'][1]}"
            )

        # Check solids range
        if not (
            config.FEATURE_RANGES["Solids"][0]
            <= solids
            <= config.FEATURE_RANGES["Solids"][1]
        ):
            errors.append(
                f"Solids must be between {config.FEATURE_RANGES['Solids'][0]} and {config.FEATURE_RANGES['Solids'][1]}"
            )

        # Check turbidity range
        if not (
            config.FEATURE_RANGES["Turbidity"][0]
            <= turbidity
            <= config.FEATURE_RANGES["Turbidity"][1]
        ):
            errors.append(
                f"Turbidity must be between {config.FEATURE_RANGES['Turbidity'][0]} and {config.FEATURE_RANGES['Turbidity'][1]}"
            )

        if errors:
            return False, "; ".join(errors)

        return True, ""

    def create_feature_vector(self, ph, solids, turbidity):
        """
        Create complete feature vector using measured values and optimal values

        Args:
            ph (float): pH value
            solids (float): Total dissolved solids
            turbidity (float): Turbidity value

        Returns:
            np.array: Complete feature vector
        """
        # Create feature dictionary
        features = {}

        # Set measured values
        features["ph"] = ph
        features["Solids"] = solids
        features["Turbidity"] = turbidity

        # Set optimal values for constrained features
        for feature in config.CONSTRAINED_FEATURES:
            features[feature] = config.OPTIMAL_VALUES[feature]

        # Create feature vector in correct order
        feature_vector = np.array(
            [features[feature] for feature in config.ALL_FEATURES]
        )

        return feature_vector.reshape(1, -1)

    def predict(self, ph, solids, turbidity, return_probability=False):
        """
        Make prediction using constrained approach

        Args:
            ph (float): pH value
            solids (float): Total dissolved solids
            turbidity (float): Turbidity value
            return_probability (bool): Whether to return probability scores

        Returns:
            dict: Prediction result
        """
        if not self.is_ready:
            return {
                "success": False,
                "error": "Predictor not ready. Please ensure model and scaler are loaded.",
            }

        # Validate input
        is_valid, error_msg = self.validate_input(ph, solids, turbidity)
        if not is_valid:
            return {"success": False, "error": f"Input validation failed: {error_msg}"}

        try:
            # Create feature vector
            feature_vector = self.create_feature_vector(ph, solids, turbidity)

            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)

            # Make prediction
            prediction = self.model.predict(feature_vector_scaled)[0]

            result = {
                "success": True,
                "prediction": int(prediction),
                "potability": "Potable" if prediction == 1 else "Not Potable",
                "input_values": {"ph": ph, "solids": solids, "turbidity": turbidity},
                "optimal_values_used": config.OPTIMAL_VALUES,
            }

            # Add probability if requested
            if return_probability:
                probabilities = self.model.predict_proba(feature_vector_scaled)[0]
                result["probability"] = {
                    "not_potable": float(probabilities[0]),
                    "potable": float(probabilities[1]),
                }

            return result

        except Exception as e:
            return {"success": False, "error": f"Prediction failed: {str(e)}"}

    def predict_batch(self, measurements, return_probability=False):
        """
        Make predictions for multiple measurements

        Args:
            measurements (list): List of dictionaries with 'ph', 'solids', 'turbidity' keys
            return_probability (bool): Whether to return probability scores

        Returns:
            list: List of prediction results
        """
        results = []

        for i, measurement in enumerate(measurements):
            try:
                ph = measurement["ph"]
                solids = measurement["solids"]
                turbidity = measurement["turbidity"]

                result = self.predict(ph, solids, turbidity, return_probability)
                result["measurement_index"] = i
                results.append(result)

            except KeyError as e:
                results.append(
                    {
                        "success": False,
                        "error": f"Missing required key in measurement {i}: {e}",
                        "measurement_index": i,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "success": False,
                        "error": f"Error processing measurement {i}: {str(e)}",
                        "measurement_index": i,
                    }
                )

        return results

    def get_feature_importance_for_measured_features(self):
        """
        Get feature importance for the measured features

        Returns:
            dict: Feature importance for measured features
        """
        if not self.is_ready:
            return None

        if hasattr(self.model, "feature_importances_"):
            importance_dict = {}
            for i, feature in enumerate(config.ALL_FEATURES):
                if feature in config.MEASURED_FEATURES:
                    importance_dict[feature] = float(self.model.feature_importances_[i])

            return importance_dict

        return None


def demo_prediction():
    """
    Demonstrate constrained prediction with sample data
    """
    print("=== Constrained Prediction Demo ===")

    # Initialize predictor
    predictor = ConstrainedPredictor()

    if not predictor.is_ready:
        print("Error: Predictor not ready. Please ensure model is trained first.")
        return

    # Sample measurements (only pH, Solids, Turbidity)
    sample_measurements = [
        {"ph": 7.0, "solids": 20000, "turbidity": 3.0},
        {"ph": 6.5, "solids": 15000, "turbidity": 2.0},
        {"ph": 8.5, "solids": 25000, "turbidity": 5.0},
        {"ph": 7.5, "solids": 18000, "turbidity": 1.5},
    ]

    print("Making predictions for sample measurements...")

    # Single prediction
    print("\n--- Single Prediction ---")
    result = predictor.predict(7.0, 20000, 3.0, return_probability=True)
    print(f"Result: {result}")

    # Batch prediction
    print("\n--- Batch Prediction ---")
    batch_results = predictor.predict_batch(
        sample_measurements, return_probability=True
    )

    for result in batch_results:
        if result["success"]:
            print(
                f"Measurement {result['measurement_index']}: {result['potability']} "
                f"(pH: {result['input_values']['ph']}, "
                f"Solids: {result['input_values']['solids']}, "
                f"Turbidity: {result['input_values']['turbidity']})"
            )
        else:
            print(
                f"Measurement {result['measurement_index']}: Error - {result['error']}"
            )

    # Feature importance
    print("\n--- Feature Importance for Measured Features ---")
    importance = predictor.get_feature_importance_for_measured_features()
    if importance:
        for feature, imp in importance.items():
            print(f"{feature}: {imp:.4f}")


if __name__ == "__main__":
    demo_prediction()
