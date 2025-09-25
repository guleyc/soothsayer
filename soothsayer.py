"""
Optimized Advanced Multi-Method Time Series Forecasting System
"""

import pandas as pd
import numpy as np
import warnings
import os
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['JOBLIB_START_METHOD'] = 'spawn'

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid, TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

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
        self.model_dir = "trained_models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.ml_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0, max_iter=2000),
            'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'Extra Trees': ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf', C=100, gamma='scale'),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42) 
        }
        
        self.param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [10, 20]
            },
            'Support Vector Regression': {
                'C': [10, 100],
                'gamma': ['scale', 'auto']
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
            
            self.data = pd.read_csv(self.data_file)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            
            date_col = None
            for col in self.data.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_col = col
                    break
            
            if date_col is None:
                date_col = self.data.columns[0]
            
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
            
            self.data.set_index(date_col, inplace=True)
            self.data.sort_index(inplace=True)
            
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.target_column = numeric_cols[0] if numeric_cols else None
            self.feature_columns = numeric_cols
            
            if self.target_column is None:
                print("Error: No numeric columns found!")
                return False
            
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
            
            self.data = self.data.ffill().bfill()
            
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
            if len(ts_data) < 60:
                return None
            
            for period in [7, 30, 365]:
                if len(ts_data) >= 2 * period:
                    try:
                        stl = STL(ts_data, period=period, robust=True)
                        result = stl.fit()

                        seasonal_strength = result.seasonal.std() / ts_data.std()
                        if seasonal_strength > 0.1:  
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
            tscv = TimeSeriesSplit(n_splits=3)

            grid_search = GridSearchCV(
                model, 
                self.param_grids[model_name], 
                cv=tscv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
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
            
            important_lags = [1, 2, 3, 7, 14, 30]
            important_lags = [l for l in important_lags if l <= len(recent_data)]
            
            for col in self.feature_columns:
                for lag in important_lags:
                    if lag <= len(recent_data):
                        features.append(recent_data[col].iloc[-lag])
                    else:
                        features.append(recent_data[col].iloc[-1])
            
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
            
            features.extend([
                target_date.weekday(),  
                target_date.month,      
                target_date.day,        
                target_date.quarter     
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
                
                important_lags = [1, 2, 3, 7, 14, 30]
                important_lags = [l for l in important_lags if l <= len(historical_slice)]
                
                for col in self.feature_columns:
                    for lag in important_lags:
                        if lag <= len(historical_slice):
                            feature_row.append(historical_slice[col].iloc[-lag])
                        else:
                            feature_row.append(historical_slice[col].iloc[-1])
                
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
            
            if X.shape[1] > 20: 
                selector = SelectKBest(score_func=f_regression, k=min(20, X.shape[1]))
                X = selector.fit_transform(X, y)
                print(f"Selected {X.shape[1]} most important features")
            
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error preparing ML data: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    def train_ml_models_optimized(self) -> None:
            """
            Optimized model training with hyperparameter tuning and caching (save/load).
            """
            try:
                X_train, X_test, y_train, y_test = self.prepare_ml_data_optimized()
                
                if len(X_train) == 0:
                    print("No training data available for ML models")
                    return
                
                print(f"Training/Loading ML models with {len(X_train)} training samples...")
                
                for name, model in tqdm(self.ml_models.items(), desc="Training/Loading ML Models"):
                    try:
                        model_path = os.path.join(self.model_dir, f"{name.replace(' ', '_')}.joblib")
                        scaler_path = os.path.join(self.model_dir, f"{name.replace(' ', '_')}_scaler.joblib")

                        if os.path.exists(model_path):
                            print(f"Loading cached model for {name}...")
                            self.models[name] = joblib.load(model_path)
                            if os.path.exists(scaler_path):
                                self.scalers[name] = joblib.load(scaler_path)
                            continue 

                        print(f"Training {name}...")
                        
                        if name in ['Support Vector Regression', 'Neural Network']:
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            self.scalers[name] = scaler
                            
                            optimized_model = self.optimize_hyperparameters(name, model, X_train_scaled, y_train)
                            optimized_model.fit(X_train_scaled, y_train)
                            
                            if len(X_test) > 0:
                                X_test_scaled = scaler.transform(X_test)
                                y_pred = optimized_model.predict(X_test_scaled)
                            else:
                                y_pred = []
                        else:
                            optimized_model = self.optimize_hyperparameters(name, model, X_train, y_train)
                            optimized_model.fit(X_train, y_train)
                            
                            if len(X_test) > 0:
                                y_pred = optimized_model.predict(X_test)
                            else:
                                y_pred = []
                        
                        self.models[name] = optimized_model
                        
                        print(f"Saving model for {name}...")
                        joblib.dump(optimized_model, model_path)
                        if name in self.scalers:
                            joblib.dump(self.scalers[name], scaler_path)

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
                        print(f"Error training/loading {name}: {e}")
                        continue
                
            except Exception as e:
                print(f"Error in ML training: {e}")
                
    def predict_statistical_models_enhanced(self, target_date: datetime, verbose: bool = True) -> Dict[str, float]:
        """Enhanced statistical models with automatic parameter selection"""
        predictions = {}
        
        try:
            ts_data = self.data[self.target_column].dropna()
            
            if len(ts_data) < 10:
                return predictions
            
            if ts_data.index.freq is None:
                ts_data = ts_data.asfreq('D', method='ffill')
            
            last_date = ts_data.index[-1]
            periods_ahead = max(1, (target_date - last_date).days)
            
            try:
                x = np.arange(len(ts_data))
                y = ts_data.values
                coeffs = np.polyfit(x, y, 1)
                trend_pred = coeffs[0] * (len(ts_data) + periods_ahead - 1) + coeffs[1]
                predictions['Linear Trend'] = trend_pred
                if verbose:
                    print(f"Linear Trend: {trend_pred:.4f}")
            except Exception as e:
                print(f"Error with Linear Trend: {e}")
            
            try:
                seasonal_period = self.detect_seasonality(ts_data)
                if seasonal_period and len(ts_data) >= seasonal_period:
                    seasonal_pred = ts_data.iloc[-seasonal_period]
                else:
                    seasonal_pred = ts_data.iloc[-1]
                predictions['Seasonal Naive'] = seasonal_pred
                if verbose:
                    print(f"Seasonal Naive: {seasonal_pred:.4f}")
            except Exception as e:
                print(f"Error with Seasonal Naive: {e}")
            
            try:
                window = min(30, len(ts_data) // 4, len(ts_data))
                if window > 0:
                    ma_pred = ts_data.tail(window).mean()
                    predictions['Moving Average'] = ma_pred
                    if verbose:
                        print(f"Moving Average: {ma_pred:.4f}")
            except Exception as e:
                print(f"Error with Moving Average: {e}")
            
            try:
                if len(ts_data) >= 10:
                    seasonal_period = self.detect_seasonality(ts_data)
                    
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
                    if verbose:
                        print(f"Exponential Smoothing: {exp_pred:.4f}")
            except Exception as e:
                print(f"Error with Exponential Smoothing: {e}")
            
            try:
                if len(ts_data) >= 20:
                    if HAS_AUTO_ARIMA:
                        model = auto_arima(
                            ts_data, 
                            seasonal=True, 
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore',
                            max_p=3, max_q=3, max_d=2,
                            n_jobs=-1
                        )
                        arima_pred = model.predict(n_periods=periods_ahead)[-1]
                    else:
                        model = ARIMA(ts_data, order=(1, 1, 1))
                        fitted_model = model.fit()
                        arima_pred = fitted_model.forecast(periods_ahead).iloc[-1]
                    
                    predictions['ARIMA'] = arima_pred
                    if verbose:
                        print(f"ARIMA: {arima_pred:.4f}")
            except Exception as e:
                print(f"Error with ARIMA: {e}")
            
            if HAS_PROPHET:
                try:
                    if len(ts_data) >= 30:
                        prophet_df = pd.DataFrame({
                            'ds': ts_data.index,
                            'y': ts_data.values
                        })

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
                        if verbose:
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
                    
                    weight = 1.0 / (0.7 * mae + 0.3 * rmse + 1e-6)
                    weights[model_name] = weight
                else:
                    weights[model_name] = 0.1
            
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            else:
                n_models = len(predictions)
                weights = {model: 1.0/n_models for model in predictions.keys()}
            
            performance_threshold = np.percentile(list(self.recent_performance.values()), 75) if self.recent_performance else float('inf')
            filtered_weights = {}
            
            for model_name, weight in weights.items():
                model_performance = self.recent_performance.get(model_name, float('inf'))
                if model_performance <= performance_threshold or len(filtered_weights) < 3: 
                    filtered_weights[model_name] = weight
            
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
        
    def predict_ml_models(self, target_date: datetime, verbose: bool = True) -> Dict[str, float]:
        """Make predictions using machine learning models"""
        predictions = {}
        
        try:
            features = self.create_enhanced_features(target_date)
            
            if features.shape[0] == 0 or features.shape[1] == 0:
                return predictions
            
            for name, model in self.models.items():
                try:
                    if name in self.scalers:
                        features_scaled = self.scalers[name].transform(features)
                        pred = model.predict(features_scaled)[0]
                    else:
                        pred = model.predict(features)[0]
                    
                    predictions[name] = pred
                    if verbose:
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
            freq = self.data.index.freq
            if freq:
                if 'D' in str(freq):
                    lookback_days = 28
                elif 'W' in str(freq):
                    lookback_days = 12  
                elif 'M' in str(freq):
                    lookback_days = 6   
                elif 'H' in str(freq):
                    lookback_days = 168 
            
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
            
            original_data = self.data.copy()
            self.data = train_data
            
            self.models = {}
            self.scalers = {}
            self.train_ml_models_optimized()
            
            for test_date, test_row in tqdm(test_data.iterrows(), total=len(test_data), desc="Calculating Recent Performance"):
                actual_value = test_row[self.target_column]
                
                ml_predictions = self.predict_ml_models(test_date, verbose=False) 
                statistical_predictions = self.predict_statistical_models_enhanced(test_date, verbose=False) 
                
                all_predictions = {**ml_predictions, **statistical_predictions}
                
                for model_name, predicted_value in all_predictions.items():
                    if model_name not in recent_performance:
                        recent_performance[model_name] = []
                    
                    error = abs(predicted_value - actual_value)
                    recent_performance[model_name].append(error)
            
            self.data = original_data
            
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
        
    def make_range_prediction(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Make predictions for a date range and create comprehensive visualizations"""
        try:
            start_dt = self._parse_date(start_date)
            end_dt = self._parse_date(end_date)
            
            if start_dt is None or end_dt is None:
                return {'error': 'Invalid date format'}
            
            if start_dt >= end_dt:
                return {'error': 'Start date must be before end date'}
            
            print(f"\nMaking range predictions from {start_dt} to {end_dt}...")
            
            if self.data.index.freq:
                freq = self.data.index.freq
            else:
                freq = 'D'
            
            date_range = pd.date_range(start=start_dt, end=end_dt, freq=freq)
            
            if len(date_range) > 365:
                print(f"Warning: Large date range ({len(date_range)} periods). This may take some time.")
            
            if not self.recent_performance:
                print("Calculating recent model performance...")
                self.calculate_recent_performance()
            
            if not self.models:
                self.train_ml_models_optimized()
            
            predictions_data = {
                'dates': [],
                'final_predictions': [],
                'confidence_lower': [],
                'confidence_upper': [],
                'individual_predictions': {},
                'model_weights': {}
            }
            
            print(f"Making predictions for {len(date_range)} dates...")
            
            for date in tqdm(date_range, desc="Range Prediction"):
                try:
                    ml_predictions = self.predict_ml_models(date, verbose=False)
                    statistical_predictions = self.predict_statistical_models_enhanced(date, verbose=False)
                    
                    all_predictions = {**ml_predictions, **statistical_predictions}
                    
                    if not all_predictions:
                        continue
                    
                    filtered_predictions = all_predictions                    
                    weights = self.calculate_dynamic_weights_enhanced(filtered_predictions)
                    weighted_prediction = sum(pred * weights.get(model, 0) for model, pred in filtered_predictions.items())
                    
                    pred_values = list(filtered_predictions.values())
                    std_pred = np.std(pred_values)
                    margin_error = 1.96 * std_pred
                    
                    predictions_data['dates'].append(date)
                    predictions_data['final_predictions'].append(weighted_prediction)
                    predictions_data['confidence_lower'].append(weighted_prediction - margin_error)
                    predictions_data['confidence_upper'].append(weighted_prediction + margin_error)
                    
                    if len(predictions_data['dates']) == 1:
                        predictions_data['individual_predictions'] = filtered_predictions
                        predictions_data['model_weights'] = weights
                    
                except Exception as e:
                    print(f"Error predicting for {date}: {e}")
                    continue
            
            if not predictions_data['dates']:
                return {'error': 'No predictions could be made for the given date range'}
            
            self._plot_range_predictions(predictions_data)
            
            final_preds = np.array(predictions_data['final_predictions'])
            summary_stats = {
                'mean_prediction': np.mean(final_preds),
                'median_prediction': np.median(final_preds),
                'std_prediction': np.std(final_preds),
                'min_prediction': np.min(final_preds),
                'max_prediction': np.max(final_preds),
                'total_periods': len(final_preds)
            }
            
            return {
                'start_date': start_date,
                'end_date': end_date,
                'predictions_data': predictions_data,
                'summary_statistics': summary_stats,
                'model_performance': self.recent_performance
            }
            
        except Exception as e:
            return {'error': f'Range prediction error: {e}'}
        
    def _plot_range_predictions(self, predictions_data: Dict) -> None:
        """Create comprehensive visualization for range predictions"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
            fig.suptitle(f'Range Prediction Analysis ({predictions_data["dates"][0].strftime("%Y-%m-%d")} to {predictions_data["dates"][-1].strftime("%Y-%m-%d")})', 
                        fontsize=16, fontweight='bold')
            
            dates = predictions_data['dates']
            final_preds = predictions_data['final_predictions']
            conf_lower = predictions_data['confidence_lower']
            conf_upper = predictions_data['confidence_upper']
            
            recent_data = self.data[self.target_column].tail(min(100, len(self.data)))
            ax1.plot(recent_data.index, recent_data.values, 'b-', linewidth=2, 
                    label=f'Historical {self.target_column}', alpha=0.8)
            
            ax1.plot(dates, final_preds, 'r-', linewidth=2, label='Predictions', alpha=0.8)
            ax1.fill_between(dates, conf_lower, conf_upper, color='red', alpha=0.2, 
                            label='95% Confidence Interval')
            
            if dates:
                ax1.axvline(x=dates[0], color='gray', linestyle='--', alpha=0.5, 
                        label='Prediction Start')
            
            ax1.set_title('Historical Data and Range Predictions', fontweight='bold')
            ax1.set_ylabel(self.target_column)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            ax2.plot(dates, final_preds, 'g-', linewidth=2, marker='o', markersize=3, alpha=0.7)
            ax2.fill_between(dates, conf_lower, conf_upper, color='green', alpha=0.2)
            
            x_numeric = np.arange(len(final_preds))
            z = np.polyfit(x_numeric, final_preds, 1)
            p = np.poly1d(z)
            ax2.plot(dates, p(x_numeric), "r--", alpha=0.8, 
                    label=f'Trend (slope: {z[0]:.4f})')
            
            ax2.set_title('Prediction Trend Analysis', fontweight='bold')
            ax2.set_ylabel('Predicted Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            if predictions_data.get('model_weights'):
                weights = predictions_data['model_weights']
                significant_weights = {k: v for k, v in weights.items() if v > 0.01}
                
                if len(significant_weights) < len(weights):
                    other_weight = sum(v for k, v in weights.items() if v <= 0.01)
                    if other_weight > 0:
                        significant_weights['Others'] = other_weight
                
                if significant_weights:
                    wedges, texts, autotexts = ax3.pie(significant_weights.values(), 
                                                    labels=significant_weights.keys(),
                                                    autopct='%1.1f%%', startangle=90)
                    ax3.set_title('Model Weights Distribution', fontweight='bold')
                    
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
            else:
                ax3.text(0.5, 0.5, 'Model weights\nnot available', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Model Weights', fontweight='bold')
            
            final_preds_array = np.array(final_preds)
            
            ax4.hist(final_preds_array, bins=min(20, len(final_preds)//3), 
                    alpha=0.7, edgecolor='black', color='skyblue')
            ax4.axvline(x=np.mean(final_preds_array), color='red', linestyle='-', 
                    linewidth=2, label=f'Mean: {np.mean(final_preds_array):.3f}')
            ax4.axvline(x=np.median(final_preds_array), color='orange', linestyle='--', 
                    linewidth=2, label=f'Median: {np.median(final_preds_array):.3f}')
            
            ax4.set_title('Prediction Distribution', fontweight='bold')
            ax4.set_xlabel('Predicted Value')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            stats_text = f'Std: {np.std(final_preds_array):.3f}\n'
            stats_text += f'Min: {np.min(final_preds_array):.3f}\n'
            stats_text += f'Max: {np.max(final_preds_array):.3f}\n'
            stats_text += f'Range: {np.ptp(final_preds_array):.3f}'
            
            ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), 
                    fontsize=9, verticalalignment='top')
            
            plt.tight_layout()
            plt.show()
            
            self._plot_detailed_trend(dates, final_preds, conf_lower, conf_upper)
            
        except Exception as e:
            print(f"Error creating range prediction plot: {e}")
            import traceback
            print(traceback.format_exc())

    def _plot_detailed_trend(self, dates, predictions, conf_lower, conf_upper):
        """Create detailed trend analysis plot"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
            fig.suptitle('Detailed Trend Analysis', fontsize=16, fontweight='bold')
            
            ax1.plot(dates, predictions, 'b-', linewidth=2, label='Predictions', alpha=0.8)
            ax1.fill_between(dates, conf_lower, conf_upper, color='blue', alpha=0.2, 
                            label='95% Confidence Interval')
            
            if len(predictions) >= 7:
                window = min(7, len(predictions)//3)
                moving_avg = pd.Series(predictions).rolling(window=window).mean()
                ax1.plot(dates, moving_avg, 'r--', linewidth=2, alpha=0.8, 
                        label=f'{window}-period Moving Average')
            
            ax1.set_title('Prediction Timeline with Confidence Bands', fontweight='bold')
            ax1.set_ylabel('Predicted Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            if len(predictions) > 1:
                changes = np.diff(predictions)
                change_dates = dates[1:]
                
                colors = ['red' if x < 0 else 'green' for x in changes]
                ax2.bar(change_dates, changes, color=colors, alpha=0.7, width=1)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                ax2.set_title('Period-to-Period Changes', fontweight='bold')
                ax2.set_ylabel('Change in Predicted Value')
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
                
                mean_change = np.mean(changes)
                std_change = np.std(changes)
                
                stats_text = f'Avg Change: {mean_change:.4f}\n'
                stats_text += f'Std Change: {std_change:.4f}\n'
                stats_text += f'Positive Changes: {sum(1 for x in changes if x > 0)}/{len(changes)}'
                
                ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), 
                        fontsize=10, verticalalignment='top')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating detailed trend plot: {e}")
    
    def make_prediction(self, target_date: str) -> Dict[str, Any]:
        """Make comprehensive prediction with all optimizations"""
        try:
            target_dt = self._parse_date(target_date)
            if target_dt is None:
                return {'error': 'Invalid date format'}

            print(f"\nMaking predictions for {target_dt}...")

            if not self.recent_performance:
                print("Calculating recent model performance...")
                self.calculate_recent_performance()

            if not self.models:
                self.train_ml_models_optimized()

            ml_predictions = self.predict_ml_models(target_dt)
            statistical_predictions = self.predict_statistical_models_enhanced(target_dt)

            all_predictions = {**ml_predictions, **statistical_predictions}

            if not all_predictions:
                return {'error': 'No predictions could be made'}

            print(f"Applying robust outlier detection (MAD) to {len(all_predictions)} raw predictions...")
            pred_series = pd.Series(list(all_predictions.values()))
            median = pred_series.median()
            mad = (pred_series - median).abs().median()

            filtered_predictions = {}

            if mad > 0:
                z_score_threshold = 2.0

                lower_bound = median - z_score_threshold * mad / 0.6745
                upper_bound = median + z_score_threshold * mad / 0.6745

                for model, pred in all_predictions.items():
                    if lower_bound <= pred <= upper_bound:
                        filtered_predictions[model] = pred
                    else:
                        print(f"Robust outlier filter removed: {model} -> {pred:.2f}")
            else:
                filtered_predictions = all_predictions

            if not filtered_predictions:
                print("Warning: Outlier filter removed all predictions. Using raw predictions.")
                filtered_predictions = all_predictions
            else:
                print(f"{len(filtered_predictions)} predictions remain after filtering.")

            weights = self.calculate_dynamic_weights_enhanced(filtered_predictions)
            self.model_weights = weights

            weighted_prediction = sum(pred * weights.get(model, 0) for model, pred in filtered_predictions.items())

            pred_values = list(all_predictions.values())
            mean_pred = np.mean(pred_values)
            median_pred = np.median(pred_values)
            std_pred = np.std(pred_values)
            min_pred = np.min(pred_values)
            max_pred = np.max(pred_values)

            z_score = 1.96
            margin_error = z_score * std_pred
            conf_lower = weighted_prediction - margin_error
            conf_upper = weighted_prediction + margin_error

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

            if filtered_predictions:
                self._plot_single_prediction(
                    target_dt, filtered_predictions, weights, weighted_prediction,
                    conf_lower, conf_upper, actual_comparison
                )

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
                    'models_used': len(filtered_predictions)
                },
                'individual_predictions': filtered_predictions,
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
    
    def _plot_single_prediction(self, target_date: datetime, predictions: Dict[str, float], 
                           weights: Dict[str, float], weighted_pred: float,
                           conf_lower: float, conf_upper: float, actual_comparison: dict = None) -> None:
        """
        Plot individual prediction results for OptimizedTimeSeriesForecaster
        """
        try:
            plt.style.use('default')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Prediction Analysis for {target_date.strftime("%Y-%m-%d")}', 
                        fontsize=16, fontweight='bold')
            
            recent_data = self.data[self.target_column].tail(min(60, len(self.data)))
            ax1.plot(recent_data.index, recent_data.values, 'b-', linewidth=2, label='Historical Data', alpha=0.7)
            
            ax1.axvline(x=target_date, color='red', linestyle='--', alpha=0.5, label='Prediction Date')
            ax1.scatter([target_date], [weighted_pred], color='red', s=100, zorder=5, 
                    label=f'Prediction: {weighted_pred:.2f}')
            
            ax1.fill_between([target_date, target_date], [conf_lower, conf_lower], [conf_upper, conf_upper], 
                            color='red', alpha=0.2, label=f'95% CI: [{conf_lower:.2f}, {conf_upper:.2f}]')
            
            if actual_comparison:
                ax1.scatter([target_date], [actual_comparison['actual_value']], 
                        color='green', s=100, marker='x', zorder=5, 
                        label=f'Actual: {actual_comparison["actual_value"]:.2f}')
            
            ax1.set_title('Historical Data and Prediction', fontweight='bold')
            ax1.set_ylabel(self.target_column)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            model_names = list(predictions.keys())
            pred_values = list(predictions.values())
            
            colors = plt.cm.tab20(np.linspace(0, 1, len(model_names)))
            
            bars = ax2.barh(model_names, pred_values, color=colors)
            ax2.axvline(x=weighted_pred, color='red', linestyle='-', linewidth=3, 
                    label=f'Weighted Avg: {weighted_pred:.2f}')
            
            if actual_comparison:
                ax2.axvline(x=actual_comparison['actual_value'], color='green', linestyle='-', 
                        linewidth=3, label=f'Actual: {actual_comparison["actual_value"]:.2f}')
            
            ax2.set_title('Individual Model Predictions', fontweight='bold')
            ax2.set_xlabel('Predicted Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, pred_values):
                width = bar.get_width()
                ax2.text(width + (max(pred_values) - min(pred_values)) * 0.01, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{value:.2f}', va='center', fontsize=9)
            
            significant_weights = {k: v for k, v in weights.items() if v > 0.01}
            if len(significant_weights) < len(weights):
                other_weight = sum(v for k, v in weights.items() if v <= 0.01)
                if other_weight > 0:
                    significant_weights['Others'] = other_weight
            
            if significant_weights:
                wedges, texts, autotexts = ax3.pie(significant_weights.values(), 
                                                labels=significant_weights.keys(),
                                                autopct='%1.1f%%', startangle=90)
                ax3.set_title('Model Weights Distribution', fontweight='bold')
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            
            if self.recent_performance:
                perf_models = [m for m in model_names if m in self.recent_performance]
                if perf_models:
                    perf_values = [self.recent_performance[m] for m in perf_models]
                    
                    bars = ax4.bar(range(len(perf_models)), perf_values, 
                                color=colors[:len(perf_models)])
                    ax4.set_title('Recent Model Performance\n(MAE - Lower is Better)', fontweight='bold')
                    ax4.set_ylabel('Mean Absolute Error')
                    ax4.set_xticks(range(len(perf_models)))
                    ax4.set_xticklabels(perf_models, rotation=45, ha='right')
                    ax4.grid(True, alpha=0.3)
                    
                    for bar, value in zip(bars, perf_values):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2, 
                                height + max(perf_values) * 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                else:
                    ax4.text(0.5, 0.5, 'No recent performance\ndata available', 
                            ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                    ax4.set_title('Model Performance', fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'No performance data\navailable yet.\nRun prediction first.', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Model Performance', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating prediction plot: {e}")
            import traceback
            print(traceback.format_exc())

    def run_backtest(self, test_split: float = 0.3) -> Dict[str, Any]:
        """
        Run backtest with plotting functionality
        """
        try:
            print("\nRunning backtest...")
            
            split_index = int(len(self.data) * (1 - test_split))
            train_data = self.data.iloc[:split_index]
            test_data = self.data.iloc[split_index:]
            
            if len(test_data) == 0:
                return {'error': 'Not enough data for backtesting'}
            
            original_data = self.data.copy()
            self.data = train_data
            
            self.models = {}
            self.scalers = {}
            self.recent_performance = {}
            
            self.calculate_recent_performance()
            self.train_ml_models_optimized()
            
            predictions = []
            actuals = []
            individual_errors = {}
            
            print(f"Testing on {len(test_data)} data points...")
            
            for i, (date, row) in enumerate(tqdm(test_data.iterrows(), total=len(test_data), desc="Running Backtest")):
                try:
                    pred_result = self.make_prediction(date.strftime('%Y-%m-%d'))
                    if 'final_prediction' in pred_result:
                        predictions.append(pred_result['final_prediction'])
                        actuals.append(row[self.target_column])
                        
                        actual_val = row[self.target_column]
                        for model_name, pred_val in pred_result['individual_predictions'].items():
                            if model_name not in individual_errors:
                                individual_errors[model_name] = []
                            individual_errors[model_name].append(abs(pred_val - actual_val))
                    
                    if (i + 1) % max(1, len(test_data) // 10) == 0:
                        print(f"Backtest progress: {i + 1}/{len(test_data)}")
                        
                except Exception as e:
                    print(f"Error in backtest prediction for {date}: {e}")
                    continue
            
            self.data = original_data
            
            if len(predictions) == 0:
                return {'error': 'No successful predictions made during backtest'}
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            r2 = r2_score(actuals, predictions)
            
            model_performance = {}
            for model_name, errors in individual_errors.items():
                if errors:
                    model_performance[model_name] = {
                        'MAE': np.mean(errors),
                        'Count': len(errors)
                    }
            
            result = {
                'test_period': f"{test_data.index[0]} to {test_data.index[-1]}",
                'predictions_made': len(predictions),
                'metrics': {
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'R²': r2
                },
                'predictions': predictions.tolist(),
                'actuals': actuals.tolist(),
                'model_performance': model_performance
            }
            
            self._plot_backtest_results(test_data.index[:len(predictions)], 
                                    predictions, actuals, result['metrics'], model_performance)
            
            return result
            
        except Exception as e:
            if 'original_data' in locals():
                self.data = original_data
            return {'error': f'Backtest error: {e}'}

    def _plot_backtest_results(self, dates, predictions, actuals, metrics, model_performance):
        """Plot backtest results for OptimizedTimeSeriesForecaster"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Backtest Results Analysis', fontsize=16, fontweight='bold')
            
            ax1.plot(dates, actuals, 'b-', label='Actual', linewidth=2, alpha=0.8)
            ax1.plot(dates, predictions, 'r--', label='Predicted', linewidth=2, alpha=0.8)
            ax1.fill_between(dates, actuals, predictions, alpha=0.2, color='gray', label='Error')
            ax1.set_title('Predictions vs Actuals Over Time')
            ax1.set_ylabel(self.target_column)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax1.tick_params(axis='x', rotation=45)
            
            ax2.scatter(actuals, predictions, alpha=0.6, s=50, color='blue')
            
            min_val = min(min(actuals), min(predictions))
            max_val = max(max(actuals), max(predictions))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
            
            ax2.set_xlabel('Actual Values')
            ax2.set_ylabel('Predicted Values')
            ax2.set_title('Predicted vs Actual Values')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            metrics_text = f'R² = {metrics["R²"]:.3f}\nMAE = {metrics["MAE"]:.3f}\nRMSE = {metrics["RMSE"]:.3f}\nMAPE = {metrics["MAPE"]:.1f}%'
            ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), 
                    fontsize=10, verticalalignment='top')
            
            errors = np.array(predictions) - np.array(actuals)
            ax3.hist(errors, bins=min(20, len(errors)//2), alpha=0.7, edgecolor='black', color='skyblue')
            ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            ax3.axvline(x=np.mean(errors), color='orange', linestyle='-', linewidth=2, 
                    label=f'Mean Error: {np.mean(errors):.3f}')
            ax3.set_title('Prediction Error Distribution')
            ax3.set_xlabel('Prediction Error')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            if model_performance:
                models = list(model_performance.keys())[:10] 
                mae_values = [model_performance[m]['MAE'] for m in models]
                
                sorted_data = sorted(zip(models, mae_values), key=lambda x: x[1])
                models_sorted, mae_sorted = zip(*sorted_data) if sorted_data else ([], [])
                
                colors = plt.cm.viridis(np.linspace(0, 1, len(models_sorted)))
                bars = ax4.bar(range(len(models_sorted)), mae_sorted, color=colors)
                
                ax4.set_title('Model Performance Comparison (MAE)', fontweight='bold')
                ax4.set_ylabel('Mean Absolute Error')
                ax4.set_xlabel('Models')
                ax4.set_xticks(range(len(models_sorted)))
                ax4.set_xticklabels(models_sorted, rotation=45, ha='right')
                ax4.grid(True, alpha=0.3)
                
                for bar, value in zip(bars, mae_sorted):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2, 
                            height + max(mae_sorted) * 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                ax4.text(0.5, 0.5, 'No model performance\ndata available', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Model Performance', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating backtest plot: {e}")
            import traceback
            print(traceback.format_exc())
        
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
            print("2. Make predictions for a date range")
            print("3. Run backtest")
            print("4. Show model performance summary") 
            print("5. Show data summary")
            print("6. Recalculate recent performance")
            print("7. Exit")          
            try:
                choice = input("\nSelect an option (1-7): ").strip()
                
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
                    start_date = input("Enter start date (various formats supported): ").strip()
                    end_date = input("Enter end date (various formats supported): ").strip()
                    
                    result = self.make_range_prediction(start_date, end_date)
                    
                    if 'error' in result:
                        print(f"Error: {result['error']}")
                    else:
                        print("\n" + "="*40)
                        print("RANGE PREDICTION RESULTS")
                        print("="*40)
                        
                        stats = result['summary_statistics']
                        print(f"\nPrediction Range: {result['start_date']} to {result['end_date']}")
                        print(f"Total Periods: {stats['total_periods']}")
                        
                        print(f"\nSummary Statistics:")
                        print(f"- Mean Prediction: {stats['mean_prediction']:.4f}")
                        print(f"- Median Prediction: {stats['median_prediction']:.4f}")
                        print(f"- Std Deviation: {stats['std_prediction']:.4f}")
                        print(f"- Min Prediction: {stats['min_prediction']:.4f}")
                        print(f"- Max Prediction: {stats['max_prediction']:.4f}")
                        
                        print(f"\nGraphs have been displayed showing:")
                        print("- Historical data with range predictions")
                        print("- Prediction trend analysis")
                        print("- Model weights distribution")
                        print("- Prediction statistics and distribution")
                
                elif choice == '3':
                    test_split = input("Enter test split ratio (default 0.3): ").strip()
                    try:
                        test_split = float(test_split) if test_split else 0.3
                    except ValueError:
                        test_split = 0.3
                    
                    result = self.run_backtest(test_split)
                    
                    if 'error' in result:
                        print(f"Error: {result['error']}")
                    else:
                        print("\n" + "="*40)
                        print("BACKTEST RESULTS")
                        print("="*40)
                        print(f"Test Period: {result['test_period']}")
                        print(f"Predictions Made: {result['predictions_made']}")
                        
                        metrics = result['metrics']
                        print(f"\nEnsemble Performance Metrics:")
                        print(f"- MAE: {metrics['MAE']:.4f}")
                        print(f"- RMSE: {metrics['RMSE']:.4f}")
                        print(f"- MAPE: {metrics['MAPE']:.2f}%")
                        print(f"- R²: {metrics['R²']:.4f}")
                
                elif choice == '4':
                    self.show_performance_summary()
                
                elif choice == '5':
                    self.show_data_summary()
                
                elif choice == '6':
                    print("Recalculating recent performance...")
                    self.recent_performance = {}
                    self.model_weights = {}
                    self.calculate_recent_performance()
                    print("Recent performance recalculated successfully!")
                
                elif choice == '7':
                    print("Thank you for using the Optimized Forecasting System!")
                    break
                
                else:
                    print("Invalid option. Please select 1-7.")
                    
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