"""
Main execution pipeline for water quality prediction system
Coordinates training, testing, and prediction phases
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Import custom modules
import config
import data_preprocessing as dp
import model_training as mt
import constrained_prediction as cp


def train_model():
    """
    Execute the complete training pipeline
    """
    print("=" * 60)
    print("WATER QUALITY PREDICTION MODEL TRAINING")
    print("=" * 60)

    # Check if data file exists
    if not os.path.exists(config.DATA_PATH):
        print(f"Error: Data file not found at {config.DATA_PATH}")
        print("Please ensure the CSV file is in the correct location.")
        return False

    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df = dp.load_data(config.DATA_PATH)
    if df is None:
        return False

    df_clean = dp.clean_data(df)
    if df_clean is None:
        return False

    # Get data statistics
    stats = dp.get_data_statistics(df_clean)
    print(f"Dataset statistics: {stats['shape'][0]} rows, {stats['shape'][1]} columns")

    # Prepare features and target
    print("\n2. Preparing features and target...")
    X, y = dp.prepare_features_and_target(df_clean)
    if X is None or y is None:
        return False

    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = dp.split_data(X, y)

    # Scale features
    print("\n4. Scaling features...")
    X_train_scaled, X_test_scaled, scaler = dp.scale_features(X_train, X_test)

    # Train model
    print("\n5. Training Random Forest model...")
    model = mt.train_random_forest(X_train_scaled, y_train)

    # Evaluate model
    print("\n6. Evaluating model...")
    results = mt.evaluate_model(model, X_test_scaled, y_test, X_train_scaled, y_train)

    # Save model and results
    print("\n7. Saving model and results...")
    mt.save_model(model)
    mt.save_results(results)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model saved to: {config.MODEL_PATH}")
    print(f"Scaler saved to: {config.SCALER_PATH}")
    print(f"Results saved to: {config.RESULTS_PATH}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test ROC AUC: {results['test_roc_auc']:.4f}")

    return True


def test_constrained_prediction():
    """
    Test the constrained prediction functionality
    """
    print("=" * 60)
    print("TESTING CONSTRAINED PREDICTION")
    print("=" * 60)

    # Initialize predictor
    predictor = cp.ConstrainedPredictor()

    if not predictor.is_ready:
        print("Error: Predictor not ready. Please train the model first.")
        return False

    # Sample test data
    test_cases = [
        {
            "name": "Good Quality Water",
            "ph": 7.0,
            "solids": 15000,
            "turbidity": 2.0,
            "expected": "Should be potable",
        },
        {
            "name": "Poor Quality Water",
            "ph": 4.0,
            "solids": 40000,
            "turbidity": 8.0,
            "expected": "Should not be potable",
        },
        {
            "name": "Moderate Quality Water",
            "ph": 6.5,
            "solids": 25000,
            "turbidity": 4.0,
            "expected": "Uncertain",
        },
        {
            "name": "Excellent Quality Water",
            "ph": 7.5,
            "solids": 10000,
            "turbidity": 1.0,
            "expected": "Should be potable",
        },
    ]

    print("\nTesting individual predictions...")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        print(
            f"Input: pH={test_case['ph']}, Solids={test_case['solids']}, Turbidity={test_case['turbidity']}"
        )
        print(f"Expected: {test_case['expected']}")

        result = predictor.predict(
            test_case["ph"],
            test_case["solids"],
            test_case["turbidity"],
            return_probability=True,
        )

        if result["success"]:
            print(f"Prediction: {result['potability']}")
            print(
                f"Probability - Not Potable: {result['probability']['not_potable']:.4f}"
            )
            print(f"Probability - Potable: {result['probability']['potable']:.4f}")
        else:
            print(f"Error: {result['error']}")

    # Test batch prediction
    print("\n--- Batch Prediction Test ---")
    batch_data = [
        {"ph": 7.0, "solids": 20000, "turbidity": 3.0},
        {"ph": 6.5, "solids": 15000, "turbidity": 2.0},
        {"ph": 8.5, "solids": 25000, "turbidity": 5.0},
    ]

    batch_results = predictor.predict_batch(batch_data, return_probability=True)

    for result in batch_results:
        if result["success"]:
            print(
                f"Sample {result['measurement_index'] + 1}: {result['potability']} "
                f"(Confidence: {result['probability']['potable']:.4f})"
            )
        else:
            print(
                f"Sample {result['measurement_index'] + 1}: Error - {result['error']}"
            )

    # Show feature importance
    print("\n--- Feature Importance for Measured Features ---")
    importance = predictor.get_feature_importance_for_measured_features()
    if importance:
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_importance:
            print(f"{feature}: {imp:.4f}")

    # Show optimal values being used
    print("\n--- Optimal Values Used for Constrained Features ---")
    for feature, value in config.OPTIMAL_VALUES.items():
        print(f"{feature}: {value}")

    print("\n" + "=" * 60)
    print("CONSTRAINED PREDICTION TESTING COMPLETED!")
    print("=" * 60)

    return True


def interactive_prediction():
    """
    Interactive prediction mode for user input
    """
    print("=" * 60)
    print("INTERACTIVE WATER QUALITY PREDICTION")
    print("=" * 60)

    # Initialize predictor
    predictor = cp.ConstrainedPredictor()

    if not predictor.is_ready:
        print("Error: Predictor not ready. Please train the model first.")
        return False

    print("\nYou will be asked to enter values for the 3 measured parameters:")
    print("- pH (0-14)")
    print("- Total Dissolved Solids (0-50000 mg/L)")
    print("- Turbidity (0-10 NTU)")
    print("\nThe system will use optimal values for other parameters.")
    print("Type 'quit' to exit.")

    while True:
        print("\n" + "-" * 40)

        try:
            # Get pH
            ph_input = input("Enter pH value (0-14): ").strip()
            if ph_input.lower() == "quit":
                break
            ph = float(ph_input)

            # Get Solids
            solids_input = input("Enter Total Dissolved Solids (mg/L): ").strip()
            if solids_input.lower() == "quit":
                break
            solids = float(solids_input)

            # Get Turbidity
            turbidity_input = input("Enter Turbidity (NTU): ").strip()
            if turbidity_input.lower() == "quit":
                break
            turbidity = float(turbidity_input)

            # Make prediction
            result = predictor.predict(ph, solids, turbidity, return_probability=True)

            if result["success"]:
                print(f"\n🔍 PREDICTION RESULT:")
                print(f"   Water Quality: {result['potability']}")
                print(f"   Confidence: {result['probability']['potable']:.2%}")

                if result["prediction"] == 1:
                    print("   ✅ This water is predicted to be safe for consumption")
                else:
                    print("   ❌ This water is predicted to be unsafe for consumption")

                print(f"\n📊 PROBABILITY BREAKDOWN:")
                print(f"   Not Potable: {result['probability']['not_potable']:.2%}")
                print(f"   Potable: {result['probability']['potable']:.2%}")

            else:
                print(f"\n❌ Error: {result['error']}")

        except ValueError:
            print("❌ Invalid input. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")

    print("\nThank you for using the Water Quality Prediction System!")
    return True


def show_model_info():
    """
    Display information about the trained model
    """
    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)

    # Check if model files exist
    model_exists = os.path.exists(config.MODEL_PATH)
    scaler_exists = os.path.exists(config.SCALER_PATH)
    results_exist = os.path.exists(config.RESULTS_PATH)

    print(f"Model file exists: {model_exists}")
    print(f"Scaler file exists: {scaler_exists}")
    print(f"Results file exists: {results_exist}")

    if results_exist:
        try:
            with open(config.RESULTS_PATH, "r") as f:
                results = json.load(f)

            print(f"\n📊 MODEL PERFORMANCE:")
            print(f"   Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"   Test ROC AUC: {results['test_roc_auc']:.4f}")

            if results["cross_validation_scores"]:
                cv_scores = results["cross_validation_scores"]
                print(f"   CV Mean Accuracy: {sum(cv_scores)/len(cv_scores):.4f}")

            print(f"\n🔍 FEATURE IMPORTANCE:")
            feature_importance = results["feature_importance"]
            for feature_info in feature_importance[:5]:  # Top 5 features
                print(f"   {feature_info['feature']}: {feature_info['importance']:.4f}")

        except Exception as e:
            print(f"Error reading results: {e}")

    print(f"\n⚙️  CONFIGURATION:")
    print(f"   Measured Features: {', '.join(config.MEASURED_FEATURES)}")
    print(f"   Constrained Features: {', '.join(config.CONSTRAINED_FEATURES)}")
    print(f"   Model Type: Random Forest")
    print(f"   Number of Estimators: {config.RANDOM_FOREST_PARAMS['n_estimators']}")

    print(f"\n🎯 OPTIMAL VALUES FOR CONSTRAINED FEATURES:")
    for feature, value in config.OPTIMAL_VALUES.items():
        print(f"   {feature}: {value}")


def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(
        description="Water Quality Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train              # Train the model
  python main.py --test               # Test constrained prediction
  python main.py --interactive        # Interactive prediction mode
  python main.py --info               # Show model information
  python main.py --all                # Run complete pipeline
        """,
    )

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--test", action="store_true", help="Test constrained prediction"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive prediction mode"
    )
    parser.add_argument("--info", action="store_true", help="Show model information")
    parser.add_argument(
        "--all", action="store_true", help="Run complete pipeline (train + test)"
    )

    args = parser.parse_args()

    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    # Execute requested operations
    if args.train or args.all:
        success = train_model()
        if not success:
            print("Training failed. Exiting.")
            return

    if args.test or args.all:
        success = test_constrained_prediction()
        if not success:
            print("Testing failed.")

    if args.interactive:
        interactive_prediction()

    if args.info:
        show_model_info()


if __name__ == "__main__":
    main()
