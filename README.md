# soothsayer
Soothsayer: An Optimized Ensemble Time Series Forecasting System

![Time Series Banner](https://guley.com.au/upload/timeseries.jpeg)

**Soothsayer** is an advanced, optimized Python framework for time series forecasting. It leverages the power of ensemble learning by combining over a dozen statistical and machine learning models to produce highly accurate predictions.

The core philosophy of this project is to mitigate the weaknesses of any single model by adopting a "wisdom of the crowd" approach. It dynamically evaluates the recent performance of each model and assigns weights accordingly, ensuring that the final forecast is robust, reliable, and adaptive to changing data patterns.

---

## ## Key Features 🚀

* **🤖 Ensemble Modeling:** Utilizes a diverse set of models—from classical statistical methods like AutoARIMA to modern machine learning algorithms like XGBoost and LightGBM—to capture various patterns in the data.
* **⚖️ Dynamic Performance-Based Weighting:** Continuously assesses models based on their recent forecasting accuracy (MAE). Better-performing models are given higher weights in the final ensemble prediction.
* **⚙️ Efficient Hyperparameter Tuning:** Employs a fast `GridSearchCV` on a subset of the data with a reduced parameter grid, finding near-optimal hyperparameters without excessive computation time.
* **🧹 Robust Preprocessing Pipeline:**
    * **Outlier Detection:** Automatically identifies and removes outliers using the IQR method.
    * **Feature Engineering:** Creates a rich feature set including time-based features (day of the week, month, quarter) and lagged values.
    * **Intelligent Feature Selection:** Uses `SelectKBest` with f-regression to select the most relevant features, reducing model complexity and preventing overfitting.
* **🌀 Automatic Seasonality Detection:** Uses STL (Seasonal and Trend decomposition using Loess) to automatically detect the dominant seasonal period (e.g., 7, 30, or 365 days) in the data.
* **🖥️ Interactive CLI:** Provides a user-friendly command-line interface for making predictions and evaluating models.

---

## ## Models Implemented

### Statistical Models
* AutoARIMA (`pmdarima`)
* Exponential Smoothing (Holt-Winters)
* Linear Trend
* Seasonal Naive
* Moving Average
* Prophet (`Facebook Prophet`)

### Machine Learning Models
* Linear Regression & Regularized Variants (Ridge, Lasso, ElasticNet)
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM
* Support Vector Regression (SVR)
* Multi-layer Perceptron (Neural Network)

---

## ## Installation 📦

To get started with Soothsayer, clone the repository and install the required dependencies.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/guleyc/soothsayer.git](https://github.com/YOUR_USERNAME/soothsayer.git)
    cd soothsayer
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the content below, then run the pip command.

    *requirements.txt:*
    ```
    pandas
    numpy
    scikit-learn
    statsmodels
    pmdarima
    xgboost
    lightgbm
    prophet
    ```

    *Installation command:*
    ```bash
    pip install -r requirements.txt
    ```

---

## ## Usage ▶️

Ensure you have a `DATA.CSV` file in the project's root directory. The file must contain a date column as its first column.

To run the interactive forecaster, execute the following command in your terminal:

```bash
python soothsayer.py
