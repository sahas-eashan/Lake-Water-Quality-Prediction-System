# Water Quality Prediction System

A machine learning system for predicting water potability using Random Forest algorithm with constrained prediction capabilities.

## Overview

This system is designed to predict water quality using only **3 measured parameters** (pH, Total Dissolved Solids, and Turbidity) while using **optimal values** for the remaining 7 parameters. This approach allows for practical water quality assessment when only basic measurements are available.

## Features

- **Supervised Learning**: Uses Random Forest classifier trained on all 10 water quality parameters
- **Constrained Prediction**: Predicts using only 3 measured parameters (pH, Solids, Turbidity)
- **Optimal Parameter Values**: Uses WHO-standard optimal values for unmeasured parameters
- **Modular Design**: Separated into distinct modules for easy maintenance and extension
- **Interactive Mode**: Command-line interface for real-time predictions
- **Batch Processing**: Support for multiple predictions at once

## Project Structure

```
water_quality_prediction/
├── config.py                    # Configuration and constants
├── data_preprocessing.py        # Data loading and preprocessing
├── model_training.py           # Model training and evaluation
├── constrained_prediction.py   # Constrained prediction functionality
├── main.py                     # Main execution pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── water_potability.csv        # Training data (place here)
```

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your data file**:
   - Ensure `water_potability.csv` is in the correct path as specified in `config.py`
   - Update `DATA_PATH` in `config.py` if needed

## Usage

### Training the Model

```bash
python main.py --train
```

This will:
- Load and preprocess the data
- Train a Random Forest model on all 10 parameters
- Evaluate the model performance
- Save the trained model and scaler

### Testing Constrained Prediction

```bash
python main.py --test
```

This will:
- Load the trained model
- Test predictions using only pH, Solids, and Turbidity
- Show sample predictions with confidence scores

### Interactive Prediction Mode

```bash
python main.py --interactive
```

This provides a user-friendly interface to:
- Enter pH, Solids, and Turbidity values
- Get real-time predictions
- See confidence scores and probability breakdowns

### Complete Pipeline

```bash
python main.py --all
```

Runs both training and testing phases.

### Model Information

```bash
python main.py --info
```

Displays:
- Model performance metrics
- Feature importance
- Configuration details
- Optimal parameter values

## Parameters

### Measured Parameters (Required for Prediction)
- **pH**: 0-14 (acidity/alkalinity)
- **Total Dissolved Solids**: 0-50,000 mg/L
- **Turbidity**: 0-10 NTU (clarity)

### Constrained Parameters (Fixed to Optimal Values)
- **Hardness**: 150.0 mg/L (moderately hard water)
- **Chloramines**: 2.0 mg/L (safe disinfection level)
- **Sulfate**: 200.0 mg/L (well below WHO guideline)
- **Conductivity**: 400.0 μS/cm (good conductivity)
- **Organic Carbon**: 2.0 mg/L (low organic content)
- **Trihalomethanes**: 50.0 μg/L (safe level)

## Model Performance

The Random Forest model typically achieves:
- **Accuracy**: ~65-70% on test data
- **ROC AUC**: ~0.65-0.75
- **Cross-validation**: 5-fold CV for robust evaluation

## Configuration

Key settings in `config.py`:
- File paths for data and model storage
- Optimal parameter values based on WHO standards
- Model hyperparameters
- Feature definitions and ranges

## API Usage

### Programmatic Usage

```python
from constrained_prediction import ConstrainedPredictor

# Initialize predictor
predictor = ConstrainedPredictor()

# Single prediction
result = predictor.predict(
    ph=7.0, 
    solids=20000, 
    turbidity=3.0, 
    return_probability=True
)

print(f"Prediction: {result['potability']}")
print(f"Confidence: {result['probability']['potable']:.2%}")

# Batch prediction
measurements = [
    {'ph': 7.0, 'solids': 20000, 'turbidity': 3.0},
    {'ph': 6.5, 'solids': 15000, 'turbidity': 2.0}
]

batch_results = predictor.predict_batch(measurements)
```

## Data Format

The training data should be a CSV file with the following columns:
- `ph`: pH value
- `Hardness`: Water hardness
- `Solids`: Total dissolved solids
- `Chloramines`: Chloramine content
- `Sulfate`: Sulfate content
- `Conductivity`: Electrical conductivity
- `Organic_carbon`: Organic carbon content
- `Trihalomethanes`: Trihalomethane content
- `Turbidity`: Turbidity level
- `Potability`: Target variable (0 = not potable, 1 = potable)

## Technical Details

### Algorithm Choice
- **Random Forest**: Chosen for its robustness and ability to handle non-linear relationships
- **Feature Scaling**: StandardScaler for consistent feature ranges
- **Cross-validation**: 5-fold CV for performance validation

### Constrained Prediction Approach
1. Uses only 3 measured parameters as input
2. Applies optimal values for 7 constrained parameters
3. Scales the complete feature vector
4. Makes prediction using the trained model

### Model Persistence
- Model saved using joblib for efficient loading
- Scaler saved separately for consistent preprocessing
- Results saved as JSON for easy analysis

## Limitations

1. **Reduced Accuracy**: Using only 3 out of 10 parameters may reduce prediction accuracy
2. **Optimal Values**: Assumes optimal values for unmeasured parameters may not reflect real conditions
3. **Data Quality**: Performance depends on the quality and representativeness of training data
4. **Generalization**: May not perform well on water sources significantly different from training data

## Future Enhancements

- **Ensemble Methods**: Combine multiple algorithms for better performance
- **Feature Engineering**: Create derived features from the 3 measured parameters
- **Uncertainty Quantification**: Provide confidence intervals for predictions
- **Real-time Integration**: API for real-time water quality monitoring systems
- **Mobile App**: User-friendly mobile interface for field measurements

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## Contact

For questions or support, please refer to the project documentation or submit an issue.