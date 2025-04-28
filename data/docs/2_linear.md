### Time Series Forecasting with Linear Models: ARMA, ARIMA, SARIMA, and SARIMAX

#### Overview

This document summarizes the methodology, implementation, and findings from a comprehensive exploration of linear time series models, including ARMA, ARIMA, SARIMA, and SARIMAX. The project demonstrates both theoretical underpinnings and practical applications, culminating in a full forecasting pipeline and a head-to-head comparison with Facebook Prophet.

---

#### Project Objectives

- Introduce and explain the family of linear time series models (AR, MA, ARMA, ARIMA, SARIMA, SARIMAX).
- Demonstrate model identification, fitting, and diagnostics using real and simulated data.
- Build a full forecasting pipeline for demand and financial time series.
- Compare classical linear models with modern alternatives (Prophet).

---

#### Methodology

**1. Linear Process Framework**

The project begins with the definition of a linear process, where each observation is a linear combination of past white noise terms. This framework unifies AR, MA, and ARMA models and provides a foundation for more advanced models.

**2. Model Identification and Simulation**

- **AR (Autoregressive) Processes:** Simulated AR(1) data is used to illustrate autocorrelation and partial autocorrelation patterns, aiding in model identification.
- **MA (Moving Average) Processes:** Simulated MA(1) data demonstrates the distinct autocorrelation structure of moving average models.
- **ARMA Processes:** The combination of AR and MA components is explored, with discussion of stationarity and characteristic polynomials.

**3. Real-World Forecasting with ARMA**

- **Dataset:** Quarterly change in US aggregate savings (FRED).
- **EDA:** Visualization and seasonal decomposition.
- **Stationarity Testing:** Augmented Dickey-Fuller (ADF) test confirms suitability for ARMA modeling.
- **Model Selection:** Use of autocorrelation and partial autocorrelation plots, and automated order selection with `pmdarima`.
- **Forecasting:** Model is trained, validated, and evaluated with confidence intervals.

**4. Beyond ARMA: ARIMA and SARIMA**

- **ARIMA:** Introduces differencing to handle non-stationary series (e.g., Tesla stock prices). Demonstrates the impact of differencing on stationarity.
- **SARIMA:** Extends ARIMA to handle seasonality, with practical guidance on identifying seasonal orders using decomposition and ACF/PACF analysis.
- **SARIMAX:** Incorporates exogenous variables for improved forecasting when additional predictors are available.

**5. Full Pipeline Example**

- **Dataset:** Kaggle Demand Forecasting competition data.
- **Steps:** Data loading, decomposition, train/test split, autocorrelation analysis, model selection (auto_arima), diagnostics, and forecasting.
- **Evaluation:** Visual and quantitative assessment of forecast accuracy.

**6. Model Comparison: SARIMAX vs Prophet**

- **Dataset:** Nifty-50 stock market data (India).
- **Features:** Rolling statistics as exogenous variables.
- **Models:** Both SARIMAX and Prophet are trained and evaluated on the same data.
- **Results:** RMSE is computed for both models, showing comparable performance.

---

#### Key Findings

- **Model Identification:** ACF and PACF plots are essential for distinguishing between AR, MA, and mixed models.
- **Stationarity:** Differencing is crucial for handling trends and making series suitable for ARIMA-type models.
- **Seasonality:** SARIMA models effectively capture seasonal patterns, provided enough data cycles are available.
- **Exogenous Variables:** SARIMAX and Prophet both benefit from additional regressors, improving forecast accuracy.
- **Model Performance:** Classical linear models (with proper tuning) can match the performance of modern tools like Prophet for many practical problems.

---

#### Example Code Snippet

from pmdarima import auto_arima
import pandas as pd

# Load and preprocess data
df = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')
series = df['value']

# Fit auto ARIMA model
model = auto_arima(series, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(series)

# Forecast
n_periods = 12
forecast = model.predict(n_periods=n_periods)

# Plot results
import matplotlib.pyplot as plt
plt.plot(series.index, series.values, label='Actual')
plt.plot(pd.date_range(series.index[-1], periods=n_periods+1, freq='M')[1:], forecast, label='Forecast')
plt.legend()
plt.show()

---

#### References

- [ARIMA Models - Statsmodels Documentation](https://www.statsmodels.org/stable/tsa.html)
- [pmdarima Documentation](https://alkaline-ml.com/pmdarima/)
- [Time Series Analysis Textbook](https://otexts.com/fpp3/)
- [Kaggle Demand Forecasting Competition](https://www.kaggle.com/c/demand-forecasting-kernels-only)
- [Nifty-50 Stock Market Data](https://www.kaggle.com/rohanrao/nifty50-stock-market-data)

---

#### Contributors

- Data Scientist: [Your Name]
- Reviewer: [Reviewer Name]
- Date: [Project Date]

---

*For questions or suggestions, please comment below or contact the project owner.*