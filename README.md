# Soothsayer: Optimized Multi-Method Time Series Forecasting System

## Overview

**Soothsayer** is a Python-based time series forecasting system designed to deliver accurate predictions by combining classical statistical methods and modern machine learning techniques. It processes time series data from a CSV file, automatically detects seasonality, optimizes model hyperparameters, and generates weighted ensemble predictions with confidence intervals. The system is optimized for performance, making it suitable for both small and large datasets.

Key features:
- **Multiple Models**: Includes Linear Trend, Seasonal Naive, Moving Average, Exponential Smoothing, ARIMA, Prophet, and machine learning models like Random Forest, XGBoost, LightGBM, and Neural Networks.
- **Dynamic Ensemble**: Combines predictions using performance-based weights for improved accuracy.
- **Feature Engineering**: Generates lagged, statistical, and time-based features.
- **Interactive Interface**: Allows users to make predictions, view model performance, and explore data summaries.

## Installation

### Prerequisites
- Python 3.8 or higher
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `statsmodels`
- Optional libraries (for enhanced functionality): `xgboost`, `lightgbm`, `prophet`, `pmdarima`

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/guleyc/soothsayer.git
   cd soothsayer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Install optional libraries for additional models:
   ```bash
   pip install xgboost lightgbm prophet pmdarima
   ```

### Requirements File
Create a `requirements.txt` with the following:
```
pandas>=1.5.0
numpy>=1.20.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
xgboost>=1.6.0
lightgbm>=3.3.0
prophet>=1.1.0
pmdarima>=2.0.0
```

## Usage

### Input Data
Soothsayer expects a CSV file (`DATA.CSV`) with:
- At least one date/time column (first column or containing "date"/"time" in the name).
- At least one numeric column for forecasting (target variable).
- Optional additional numeric columns as features.

Example CSV format:
```csv
Date,Value,Feature1
2023-01-01,100,50
2023-01-02,105,52
...
```

### Running the System
1. Place your `DATA.CSV` file in the project directory.
2. Run the script:
   ```bash
   python soothsayer.py
   ```
3. Follow the interactive menu to:
   - Make predictions for a specific date.
   - View model performance summaries.
   - Display data summaries.
   - Recalculate recent model performance.

### Example
```bash
$ python soothsayer.py
Optimized Advanced Multi-Method Forecasting System
===================================================
Data loaded successfully. Shape: (100, 2)
Successfully parsed dates using format: %Y-%m-%d
Inferred and set frequency: D

==================================================
OPTIONS:
1. Make a prediction for a specific date
2. Show model performance summary
3. Show data summary
4. Recalculate recent performance
5. Exit

Select an option (1-5): 1
Enter target date (various formats supported): 2023-12-01
```

## Key Components

### Statistical Models
- **Linear Trend**: Fits a linear model to capture trends.
- **Seasonal Naive**: Uses past seasonal values for prediction.
- **Moving Average**: Averages recent data points.
- **Exponential Smoothing**: Models level, trend, and seasonality with exponential weights.
- **ARIMA**: Autoregressive Integrated Moving Average with automatic parameter selection.
- **Prophet**: Handles multiple seasonalities and missing data (optional).

### Machine Learning Models
- Linear Regression, Ridge, Lasso, Elastic Net
- Decision Tree, Random Forest, Extra Trees
- Gradient Boosting, XGBoost, LightGBM (optional)
- Support Vector Regression, Neural Network (MLP)

### Ensemble Prediction
Predictions are combined using dynamic weights based on recent model performance (MAE and RMSE). A 95% confidence interval is provided to quantify uncertainty.

## Customization
- **Hyperparameter Tuning**: Adjust the `param_grids` dictionary in the `OptimizedTimeSeriesForecaster` class to modify hyperparameter search spaces.
- **Feature Engineering**: Modify the `create_enhanced_features` method to add custom features.
- **Data Preprocessing**: Adjust the `load_data` method to handle specific date formats or preprocessing steps.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.
