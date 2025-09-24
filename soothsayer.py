#!/usr/bin/env python3
"""
Optimized Advanced Multi-Method Time Series Forecasting System
"""

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['JOBLIB_START_METHOD'] = 'spawn'

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.feature_selection import SelectKBest, f_regression

# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

# Optional libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    from pmdarima import auto_arima
    HAS_AUTO_ARIMA = True
except ImportError:
    HAS_AUTO_ARIMA = False


class OptimizedTimeSeriesForecaster:
    """
    Optimized time series forecasting system with faster hyperparameter tuning
    """
    
    def __init__(self, data_file: str = 'DATA.CSV'):
        self.data_file = data_file
        self.data = None
        self.target_column = None
        self.feature_columns = []
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.model_weights = {}
        self.recent_performance = {}
        self.predictions_cache = {}
        
        # Optimized model configurations with limited hyperparameter search
        self.ml_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0, max_iter=2000),
            'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),  # Reduced estimators, parallel
            'Extra Trees': ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),  # Reduced estimators
            'Support Vector Regression': SVR(kernel='rbf', C=100, gamma='scale'),  # Fixed parameters, no tuning
            'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)  # Smaller network
        }
        
        # Hyperparameter grids (reduced for speed)
        self.param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [10, 20]
            },
            'Support Vector Regression': {
                'C': [10, 100],  # Reduced grid
                'gamma': ['scale', 'auto']  # Only 2 options
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50, 25), (100, 50)],
                'alpha': [0.001, 0.01]
            }
        }
        
        if HAS_XGBOOST:
            self.ml_models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=50, 
                random_state=42, 
                n_jobs=-1,
                verbosity=0
            )
            self.param_grids['XGBoost'] = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6]
            }
        
        if HAS_LIGHTGBM:
            self.ml_models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=50, 
                random_state=42, 
                n_jobs=-1,
                verbosity=-1
            )
            self.param_grids['LightGBM'] = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6]
            }
    
    def load_data(self) -> bool:
        """Load and preprocess data with outlier detection"""
        try:
            if not os.path.exists(self.data_file):
                print(f"Error: File '{self.data_file}' not found!")
                return False
            
            # Load data
            self.data = pd.read_csv(self.data_file)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            
            # Identify date column
            date_col = None
            for col in self.data.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_col = col
                    break
            
            if date_col is None:
                date_col = self.data.columns[0]
            
            # Parse dates
            date_formats = [
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%d-%m-%Y',
                '%m-%d-%Y'
            ]
            
            parsed = False
            for fmt in date_formats:
                try:
                    self.data[date_col] = pd.to_datetime(self.data[date_col], format=fmt)
                    print(f"Successfully parsed dates using format: {fmt}")
                    parsed = True
                    break
                except (ValueError, TypeError):
                    continue
            
            if not parsed:
                self.data[date_col] = pd.to_datetime(self.data[date_col])
            
            # Set date as index
            self.data.set_index(date_col, inplace=True)
            self.data.sort_index(inplace=True)
            
            # Identify numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.target_column = numeric_cols[0] if numeric_cols else None
            self.feature_columns = numeric_cols
            
            if self.target_column is None:
                print("Error: No numeric columns found!")
                return False
            
            # Outlier detection and removal (IQR method)
            Q1 = self.data[self.target_column].quantile(0.25)
            Q3 = self.data[self.target_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = len(self.data)
            self.data = self.data[
                (self.data[self.target_column] >= lower_bound) & 
                (self.data[self.target_column] <= upper_bound)
            ]
            outliers_removed = outliers_before - len(self.data)
            
            if outliers_removed > 0:
                print(f"Removed {outliers_removed} outliers")
            
            # Fill missing values
            self.data = self.data.ffill().bfill()
            
            # Set frequency
            try:
                freq = pd.infer_freq(self.data.index[:min(100, len(self.data))])
                if freq:
                    self.data = self.data.asfreq(freq, method='ffill')
                    print(f"Inferred and set frequency: {freq}")
                else:
                    self.data = self.data.asfreq('D', method='ffill')
                    print("Could not infer frequency. Resampled to daily ('D') frequency.")
            
            except Exception as e:
                print(f"Frequency handling failed: {e}. Falling back to daily resampling.")
                self.data = self.data.asfreq('D', method='ffill')            
                print(f"Target column: {self.target_column}")
                print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def detect_seasonality(self, ts_data: pd.Series) -> int:
        """Detect seasonal period using STL decomposition"""
        try:
            if len(ts_data) < 60:  # Need enough data for seasonality detection
                return None
            
            # Try different seasonal periods
            for period in [7, 30, 365]:  # Daily, monthly, yearly
                if len(ts_data) >= 2 * period:
                    try:
                        stl = STL(ts_data, period=period, robust=True)
                        result = stl.fit()
                        
                        # Check if seasonal component is significant
                        seasonal_strength = result.seasonal.std() / ts_data.std()
                        if seasonal_strength > 0.1:  # 10% threshold
                            return period
                    except:
                        continue
            
            return None
            
        except Exception:
            return None
    
    def optimize_hyperparameters(self, model_name: str, model, X_train: np.ndarray, y_train: np.ndarray) -> any:
        """Fast hyperparameter optimization with reduced search space"""
        if model_name not in self.param_grids:
            return model
        
        try:
            print(f"Optimizing hyperparameters for {model_name}...")
            
            # Use smaller CV for speed (3-fold instead of 5)
            grid_search = GridSearchCV(
                model, 
                self.param_grids[model_name], 
                cv=3,  # Reduced CV folds
                scoring='neg_mean_absolute_error',
                n_jobs=-1,  # Use all cores
                verbose=0
            )
            
            # Use subset of data for hyperparameter tuning if dataset is large
            if len(X_train) > 500:
                subset_size = min(500, len(X_train))
                indices = np.random.choice(len(X_train), subset_size, replace=False)
                X_subset = X_train[indices]
                y_subset = y_train[indices]
            else:
                X_subset = X_train
                y_subset = y_train
            
            grid_search.fit(X_subset, y_subset)
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            print(f"Error optimizing {model_name}: {e}")
            return model
    
    def create_enhanced_features(self, target_date: datetime, lookback_window: int = 30) -> np.ndarray:
        """Create enhanced features with domain-specific additions"""
        try:
            historical_data = self.data[self.data.index < target_date]
            
            if len(historical_data) < lookback_window:
                lookback_window = len(historical_data)
            
            if lookback_window == 0:
                return np.array([]).reshape(0, -1)
            
            recent_data = historical_data.tail(lookback_window)
            features = []
            
            # Lagged features (optimized selection)
            important_lags = [1, 2, 3, 7, 14, 30]
            important_lags = [l for l in important_lags if l <= len(recent_data)]
            
            for col in self.feature_columns:
                for lag in important_lags:
                    if lag <= len(recent_data):
                        features.append(recent_data[col].iloc[-lag])
                    else:
                        features.append(recent_data[col].iloc[-1])
            
            # Statistical features
            for col in self.feature_columns:
                col_data = recent_data[col]
                if len(col_data) > 0:
                    features.extend([
                        col_data.mean(),
                        col_data.std() if len(col_data) > 1 else 0,
                        col_data.min(),
                        col_data.max(),
                        col_data.iloc[-1],
                        col_data.quantile(0.25) if len(col_data) > 1 else col_data.iloc[0],
                        col_data.quantile(0.75) if len(col_data) > 1 else col_data.iloc[0]
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0, 0, 0])
            
            # Time-based features
            features.extend([
                target_date.weekday(),  # Day of week (0-6)
                target_date.month,      # Month (1-12)
                target_date.day,        # Day of month (1-31)
                target_date.quarter     # Quarter (1-4)
            ])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Error creating features: {e}")
            return np.array([]).reshape(0, -1)
    
    def prepare_ml_data_optimized(self, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Optimized data preparation with feature selection"""
        try:
            lookback_window = min(30, len(self.data) // 10)
            X, y = [], []
            
            for i in range(lookback_window, len(self.data)):
                historical_slice = self.data.iloc[max(0, i-lookback_window):i]
                feature_row = []
                
                # Optimized lagged features
                important_lags = [1, 2, 3, 7, 14, 30]
                important_lags = [l for l in important_lags if l <= len(historical_slice)]
                
                for col in self.feature_columns:
                    for lag in important_lags:
                        if lag <= len(historical_slice):
                            feature_row.append(historical_slice[col].iloc[-lag])
                        else:
                            feature_row.append(historical_slice[col].iloc[-1])
                
                # Statistical features
                for col in self.feature_columns:
                    col_data = historical_slice[col]
                    if len(col_data) > 0:
                        feature_row.extend([
                            col_data.mean(),
                            col_data.std() if len(col_data) > 1 else 0,
                            col_data.min(),
                            col_data.max(),
                            col_data.iloc[-1],
                            col_data.quantile(0.25) if len(col_data) > 1 else col_data.iloc[0],
                            col_data.quantile(0.75) if len(col_data) > 1 else col_data.iloc[0]
                        ])
                    else:
                        feature_row.extend([0, 0, 0, 0, 0, 0, 0])
                
                # Time features
                date = self.data.index[i]
                feature_row.extend([
                    date.weekday(),
                    date.month,
                    date.day,
                    date.quarter
                ])
                
                X.append(feature_row)
                y.append(self.data.iloc[i][self.target_column])
            
            X = np.array(X)
            y = np.array(y)
            
            if len(X) == 0:
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            # Feature selection to reduce dimensionality
            if X.shape[1] > 20:  # Only if we have many features
                selector = SelectKBest(score_func=f_regression, k=min(20, X.shape[1]))
                X = selector.fit_transform(X, y)
                print(f"Selected {X.shape[1]} most important features")
            
            # Split data chronologically
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error preparing ML data: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    def train_ml_models_optimized(self) -> None:
        """Optimized model training with hyperparameter tuning"""
        try:
            X_train, X_test, y_train, y_test = self.prepare_ml_data_optimized()
            
            if len(X_train) == 0:
                print("No training data available for ML models")
                return
            
            print(f"Training ML models with {len(X_train)} training samples...")
            
            for name, model in self.ml_models.items():
                try:
                    print(f"Training {name}...")
                    
                    # Scale features for certain models
                    if name in ['Support Vector Regression', 'Neural Network']:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        self.scalers[name] = scaler
                        
                        # Optimize hyperparameters
                        optimized_model = self.optimize_hyperparameters(name, model, X_train_scaled, y_train)
                        optimized_model.fit(X_train_scaled, y_train)
                        
                        if len(X_test) > 0:
                            X_test_scaled = scaler.transform(X_test)
                            y_pred = optimized_model.predict(X_test_scaled)
                        else:
                            y_pred = []
                    else:
                        # Optimize hyperparameters for other models
                        optimized_model = self.optimize_hyperparameters(name, model, X_train, y_train)
                        optimized_model.fit(X_train, y_train)
                        
                        if len(X_test) > 0:
                            y_pred = optimized_model.predict(X_test)
                        else:
                            y_pred = []
                    
                    self.models[name] = optimized_model
                    
                    # Calculate performance metrics
                    if len(y_test) > 0 and len(y_pred) > 0:
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        r2 = r2_score(y_test, y_pred)
                        
                        self.performance_metrics[name] = {
                            'MAE': mae,
                            'RMSE': rmse,
                            'R²': r2
                        }
                        
                        print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
                    
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
        except Exception as e:
            print(f"Error in ML training: {e}")
    
    def predict_statistical_models_enhanced(self, target_date: datetime) -> Dict[str, float]:
        """Enhanced statistical models with automatic parameter selection"""
        predictions = {}
        
        try:
            ts_data = self.data[self.target_column].dropna()
            
            if len(ts_data) < 10:
                return predictions
            
            # Ensure proper frequency
            if ts_data.index.freq is None:
                ts_data = ts_data.asfreq('D', method='ffill')
            
            last_date = ts_data.index[-1]
            periods_ahead = max(1, (target_date - last_date).days)
            
            # Linear Trend
            try:
                x = np.arange(len(ts_data))
                y = ts_data.values
                coeffs = np.polyfit(x, y, 1)
                trend_pred = coeffs[0] * (len(ts_data) + periods_ahead - 1) + coeffs[1]
                predictions['Linear Trend'] = trend_pred
                print(f"Linear Trend: {trend_pred:.4f}")
            except Exception as e:
                print(f"Error with Linear Trend: {e}")
            
            # Enhanced Seasonal Naive with automatic seasonality detection
            try:
                seasonal_period = self.detect_seasonality(ts_data)
                if seasonal_period and len(ts_data) >= seasonal_period:
                    seasonal_pred = ts_data.iloc[-seasonal_period]
                else:
                    seasonal_pred = ts_data.iloc[-1]
                predictions['Seasonal Naive'] = seasonal_pred
                print(f"Seasonal Naive: {seasonal_pred:.4f}")
            except Exception as e:
                print(f"Error with Seasonal Naive: {e}")
            
            # Moving Average
            try:
                window = min(30, len(ts_data) // 4, len(ts_data))
                if window > 0:
                    ma_pred = ts_data.tail(window).mean()
                    predictions['Moving Average'] = ma_pred
                    print(f"Moving Average: {ma_pred:.4f}")
            except Exception as e:
                print(f"Error with Moving Average: {e}")
            
            # Enhanced Exponential Smoothing with automatic trend/seasonal detection
            try:
                if len(ts_data) >= 10:
                    seasonal_period = self.detect_seasonality(ts_data)
                    
                    # Determine if there's a trend
                    trend_test = adfuller(ts_data.diff().dropna())[1] < 0.05
                    
                    model = ExponentialSmoothing(
                        ts_data,
                        trend='add' if trend_test else None,
                        seasonal='add' if seasonal_period else None,
                        seasonal_periods=seasonal_period
                    )
                    fitted_model = model.fit()
                    exp_pred = fitted_model.forecast(periods_ahead).iloc[-1]
                    predictions['Exponential Smoothing'] = exp_pred
                    print(f"Exponential Smoothing: {exp_pred:.4f}")
            except Exception as e:
                print(f"Error with Exponential Smoothing: {e}")
            
            # Auto ARIMA if available, otherwise simple ARIMA
            try:
                if len(ts_data) >= 20:
                    if HAS_AUTO_ARIMA:
                        model = auto_arima(
                            ts_data, 
                            seasonal=True, 
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            max_p=3, max_q=3, max_d=2
                        )
                        arima_pred = model.predict(n_periods=periods_ahead)[-1]
                    else:
                        model = ARIMA(ts_data, order=(1, 1, 1))
                        fitted_model = model.fit()
                        arima_pred = fitted_model.forecast(periods_ahead).iloc[-1]
                    
                    predictions['ARIMA'] = arima_pred
                    print(f"ARIMA: {arima_pred:.4f}")
            except Exception as e:
                print(f"Error with ARIMA: {e}")
            
            # Enhanced Prophet
            if HAS_PROPHET:
                try:
                    if len(ts_data) >= 30:
                        prophet_df = pd.DataFrame({
                            'ds': ts_data.index,
                            'y': ts_data.values
                        })
                        
                        # Determine seasonality based on data length
                        model = Prophet(
                            daily_seasonality=len(ts_data) > 365,
                            weekly_seasonality=len(ts_data) > 90,
                            yearly_seasonality=len(ts_data) > 365,
                            changepoint_prior_scale=0.05
                        )
                        model.fit(prophet_df)
                        
                        future_df = pd.DataFrame({'ds': [target_date]})
                        forecast = model.predict(future_df)
                        prophet_pred = forecast['yhat'].iloc[0]
                        
                        predictions['Prophet'] = prophet_pred
                        print(f"Prophet: {prophet_pred:.4f}")
                except Exception as e:
                    print(f"Error with Prophet: {e}")
            
        except Exception as e:
            print(f"Error in statistical predictions: {e}")
        
        return predictions
    
    def calculate_dynamic_weights_enhanced(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Enhanced weighting with multiple metrics"""
        try:
            if not self.recent_performance:
                n_models = len(predictions)
                return {model: 1.0/n_models for model in predictions.keys()}
            
            weights = {}
            
            for model_name in predictions.keys():
                if model_name in self.recent_performance:
                    mae = self.recent_performance[model_name]
                    rmse = self.performance_metrics.get(model_name, {}).get('RMSE', mae)
                    
                    # Combined weight using both MAE and RMSE
                    weight = 1.0 / (0.7 * mae + 0.3 * rmse + 1e-6)
                    weights[model_name] = weight
                else:
                    weights[model_name] = 0.1
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            else:
                n_models = len(predictions)
                weights = {model: 1.0/n_models for model in predictions.keys()}
            
            # Apply minimum performance threshold
            performance_threshold = np.percentile(list(self.recent_performance.values()), 75) if self.recent_performance else float('inf')
            filtered_weights = {}
            
            for model_name, weight in weights.items():
                model_performance = self.recent_performance.get(model_name, float('inf'))
                if model_performance <= performance_threshold or len(filtered_weights) < 3:  # Keep at least 3 models
                    filtered_weights[model_name] = weight
            
            # Renormalize filtered weights
            total_filtered = sum(filtered_weights.values())
            if total_filtered > 0:
                filtered_weights = {k: v/total_filtered for k, v in filtered_weights.items()}
                return filtered_weights
            else:
                return weights
            
        except Exception as e:
            print(f"Error calculating enhanced weights: {e}")
            n_models = len(predictions)
            return {model: 1.0/n_models for model in predictions.keys()}
    
    # Include other necessary methods (predict_ml_models, make_prediction, etc.)
    # ... (keeping the existing methods with modifications for optimization)
    
    def predict_ml_models(self, target_date: datetime) -> Dict[str, float]:
        """Make predictions using machine learning models"""
        predictions = {}
        
        try:
            features = self.create_enhanced_features(target_date)
            
            if features.shape[0] == 0 or features.shape[1] == 0:
                print("Warning: Could not create features for ML prediction")
                return predictions
            
            for name, model in self.models.items():
                try:
                    if name in self.scalers:
                        features_scaled = self.scalers[name].transform(features)
                        pred = model.predict(features_scaled)[0]
                    else:
                        pred = model.predict(features)[0]
                    
                    predictions[name] = pred
                    print(f"{name}: {pred:.4f}")
                    
                except Exception as e:
                    print(f"Error predicting with {name}: {e}")
                    continue
            
        except Exception as e:
            print(f"Error in ML predictions: {e}")
        
        return predictions
    
    def calculate_recent_performance(self, lookback_days: int = 28) -> Dict[str, float]:
        """Calculate recent performance with adaptive lookback"""
        try:
            # Adaptive lookback based on data frequency
            freq = self.data.index.freq
            if freq:
                if 'D' in str(freq):
                    lookback_days = 28
                elif 'W' in str(freq):
                    lookback_days = 12  # 12 weeks
                elif 'M' in str(freq):
                    lookback_days = 6   # 6 months
                elif 'H' in str(freq):
                    lookback_days = 168  # 1 week in hours
            
            recent_performance = {}
            
            if len(self.data) < lookback_days + 30:
                print("Not enough data for recent performance calculation")
                return {}
            
            split_date = self.data.index[-lookback_days]
            train_data = self.data[self.data.index < split_date]
            test_data = self.data[self.data.index >= split_date]
            
            if len(test_data) == 0:
                return {}
            
            print(f"Calculating recent performance on last {len(test_data)} data points...")
            
            # Temporarily set training data
            original_data = self.data.copy()
            self.data = train_data
            
            # Train models on limited training data
            self.models = {}
            self.scalers = {}
            self.train_ml_models_optimized()
            
            # Test each model on recent data
            for test_idx, (test_date, test_row) in enumerate(test_data.iterrows()):
                actual_value = test_row[self.target_column]
                
                # Get predictions
                ml_predictions = self.predict_ml_models(test_date)
                statistical_predictions = self.predict_statistical_models_enhanced(test_date)
                
                all_predictions = {**ml_predictions, **statistical_predictions}
                
                # Calculate errors for each model
                for model_name, predicted_value in all_predictions.items():
                    if model_name not in recent_performance:
                        recent_performance[model_name] = []
                    
                    error = abs(predicted_value - actual_value)
                    recent_performance[model_name].append(error)
            
            # Restore original data
            self.data = original_data
            
            # Calculate average error for each model
            model_scores = {}
            for model_name, errors in recent_performance.items():
                if errors:
                    model_scores[model_name] = np.mean(errors)
            
            self.recent_performance = model_scores
            print("Recent performance calculated successfully")
            return model_scores
            
        except Exception as e:
            if 'original_data' in locals():
                self.data = original_data
            print(f"Error calculating recent performance: {e}")
            return {}
    
    def make_prediction(self, target_date: str) -> Dict[str, Any]:
        """Make comprehensive prediction with all optimizations"""
        try:
            target_dt = self._parse_date(target_date)
            if target_dt is None:
                return {'error': 'Invalid date format'}
            
            print(f"\nMaking predictions for {target_dt}...")
            
            # Calculate recent performance if not already done
            if not self.recent_performance:
                print("Calculating recent model performance...")
                self.calculate_recent_performance()
            
            # Train models if not already trained
            if not self.models:
                self.train_ml_models_optimized()
            
            # Get predictions from all methods
            ml_predictions = self.predict_ml_models(target_dt)
            statistical_predictions = self.predict_statistical_models_enhanced(target_dt)
            
            # Combine all predictions
            all_predictions = {**ml_predictions, **statistical_predictions}
            
            if not all_predictions:
                return {'error': 'No predictions could be made'}
            
            # Calculate enhanced dynamic weights
            weights = self.calculate_dynamic_weights_enhanced(all_predictions)
            self.model_weights = weights
            
            # Calculate weighted prediction
            weighted_prediction = sum(pred * weights.get(model, 0) for model, pred in all_predictions.items())
            
            # Calculate ensemble statistics
            pred_values = list(all_predictions.values())
            mean_pred = np.mean(pred_values)
            median_pred = np.median(pred_values)
            std_pred = np.std(pred_values)
            min_pred = np.min(pred_values)
            max_pred = np.max(pred_values)
            
            # Enhanced confidence interval calculation
            z_score = 1.96  # 95% confidence
            margin_error = z_score * std_pred
            conf_lower = weighted_prediction - margin_error
            conf_upper = weighted_prediction + margin_error
            
            # Compare with actual value if available
            actual_comparison = None
            if target_dt in self.data.index:
                actual_value = self.data.loc[target_dt, self.target_column]
                error = abs(weighted_prediction - actual_value)
                error_pct = (error / actual_value) * 100
                actual_comparison = {
                    'actual_value': actual_value,
                    'error': error,
                    'error_percentage': error_pct,
                    'is_within_ci': conf_lower <= actual_value <= conf_upper
                }
                print(f"\n*** ACTUAL VALUE COMPARISON ***")
                print(f"Actual Value: {actual_value:.4f}")
                print(f"Predicted Value: {weighted_prediction:.4f}")
                print(f"Error: {error:.4f} ({error_pct:.2f}%)")
                print(f"Within Confidence Interval: {actual_comparison['is_within_ci']}")
            
            return {
                'target_date': target_date,
                'final_prediction': weighted_prediction,
                'confidence_interval': (conf_lower, conf_upper),
                'statistics': {
                    'mean': mean_pred,
                    'median': median_pred,
                    'std': std_pred,
                    'min': min_pred,
                    'max': max_pred,
                    'models_used': len(all_predictions)
                },
                'individual_predictions': all_predictions,
                'model_weights': weights,
                'recent_performance': self.recent_performance,
                'actual_comparison': actual_comparison
            }
            
        except Exception as e:
            return {'error': f'Prediction error: {e}'}
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats"""
        date_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
            '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%d.%m.%Y', '%m.%d.%Y', '%Y.%m.%d'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        try:
            return pd.to_datetime(date_str).to_pydatetime()
        except:
            return None
    
    def run_interactive_mode(self) -> None:
        """Run the interactive forecasting interface"""
        print("Optimized Advanced Multi-Method Forecasting System")
        print("="*55)
        
        if not self.load_data():
            print("Failed to load data. Please check your DATA.CSV file.")
            return
        
        while True:
            print("\n" + "="*50)
            print("OPTIONS:")
            print("1. Make a prediction for a specific date")
            print("2. Show model performance summary")
            print("3. Show data summary")
            print("4. Recalculate recent performance")
            print("5. Exit")
            
            try:
                choice = input("\nSelect an option (1-5): ").strip()
                
                if choice == '1':
                    target_date = input("Enter target date (various formats supported): ").strip()
                    result = self.make_prediction(target_date)
                    
                    if 'error' in result:
                        print(f"Error: {result['error']}")
                    else:
                        print("\n" + "="*40)
                        print("PREDICTION RESULTS")
                        print("="*40)
                        
                        print(f"\nFINAL PREDICTION for {result['target_date']}:")
                        print(f"Predicted {self.target_column}: {result['final_prediction']:.4f}")
                        print(f"Confidence Interval (95%): ({result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f})")
                        
                        stats = result['statistics']
                        print(f"\nPrediction Statistics:")
                        print(f"- Mean: {stats['mean']:.4f}")
                        print(f"- Median: {stats['median']:.4f}")
                        print(f"- Standard Deviation: {stats['std']:.4f}")
                        print(f"- Range: {stats['min']:.4f} to {stats['max']:.4f}")
                        print(f"- Models used: {stats['models_used']}")
                        
                        print(f"\nTop 5 Weighted Models:")
                        sorted_weights = sorted(result['model_weights'].items(), key=lambda x: x[1], reverse=True)
                        for model, weight in sorted_weights[:5]:
                            pred = result['individual_predictions'].get(model, 0)
                            print(f"- {model}: {weight:.3f} (Prediction: {pred:.2f})")
                        
                        if result.get('actual_comparison'):
                            comp = result['actual_comparison']
                            print(f"\n*** ACTUAL VALUE COMPARISON ***")
                            print(f"Actual: {comp['actual_value']:.4f}")
                            print(f"Error: {comp['error']:.4f} ({comp['error_percentage']:.2f}%)")
                            print(f"Within CI: {comp['is_within_ci']}")
                
                elif choice == '2':
                    self.show_performance_summary()
                
                elif choice == '3':
                    self.show_data_summary()
                
                elif choice == '4':
                    print("Recalculating recent performance...")
                    self.recent_performance = {}
                    self.model_weights = {}
                    self.calculate_recent_performance()
                    print("Recent performance recalculated successfully!")
                
                elif choice == '5':
                    print("Thank you for using the Optimized Forecasting System!")
                    break
                
                else:
                    print("Invalid option. Please select 1-5.")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
    
    def show_performance_summary(self) -> None:
        """Display model performance summary"""
        print("\n" + "="*40)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*40)
        
        if not self.performance_metrics and not self.recent_performance:
            print("No model performance data available. Run a prediction first.")
            return
        
        if self.performance_metrics:
            print("\nTraining Performance (Machine Learning Models):")
            for model_name, metrics in self.performance_metrics.items():
                print(f"{model_name}:")
                for metric_name, value in metrics.items():
                    print(f"  - {metric_name}: {value:.4f}")
        
        if self.recent_performance:
            print(f"\nRecent Performance (MAE - Lower is Better):")
            sorted_performance = sorted(self.recent_performance.items(), key=lambda x: x[1])
            for model_name, mae in sorted_performance:
                print(f"  - {model_name}: {mae:.4f}")
        
        if self.model_weights:
            print(f"\nCurrent Model Weights:")
            sorted_weights = sorted(self.model_weights.items(), key=lambda x: x[1], reverse=True)
            for model_name, weight in sorted_weights:
                print(f"  - {model_name}: {weight:.4f} ({weight*100:.1f}%)")
    
    def show_data_summary(self) -> None:
        """Display data summary"""
        print("\n" + "="*40)
        print("DATA SUMMARY")
        print("="*40)
        
        if self.data is None:
            print("No data loaded.")
            return
        
        print(f"Data shape: {self.data.shape}")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Target column: {self.target_column}")
        print(f"Frequency: {self.data.index.freq}")
        
        if self.target_column:
            print(f"\nTarget column statistics:")
            print(self.data[self.target_column].describe())


def main():
    """Main function to run the optimized forecasting system"""
    try:
        forecaster = OptimizedTimeSeriesForecaster()
        forecaster.run_interactive_mode()
    except Exception as e:
        print(f"System error: {e}")


if __name__ == "__main__":
    main()