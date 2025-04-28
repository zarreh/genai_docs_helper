# Machine Learning Project: Time Series Analysis Basics

## Overview
This document provides an introduction to time series analysis, focusing on elementary methods and foundational concepts. It is part of a series on practical time series methods.

## Table of Contents
1. Introduction
2. Groundwork
3. Patterns in Time Series
4. Dependence and Stationarity
5. Seasonal Decomposition
6. Transformations for Stationarity
7. Autocorrelation and Partial Autocorrelation
8. References

## Introduction
Time series analysis is crucial for understanding data that is collected over time. This document covers the basics of time series analysis, including patterns, dependence, and stationarity.

## Groundwork
Time series data can be decomposed into several components:
- **Trend**: The long-term movement in the data.
- **Seasonal**: The repeating short-term cycle in the data.
- **Cyclical**: The long-term cycle in the data.
- **Irregular**: The random variation in the data.

## Patterns in Time Series
Understanding patterns in time series data is essential for accurate modeling and forecasting. Common patterns include:
- **Trend**: Long-term increase or decrease.
- **Seasonal**: Regular pattern repeating over time.
- **Cyclical**: Irregular fluctuations not of fixed period.
- **Irregular**: Random noise.

## Dependence and Stationarity
### Dependence
Dependence in time series refers to the relationship between observations at different times. It is crucial for modeling and forecasting.

### Stationarity
A stationary time series has properties that do not depend on the time at which the series is observed. Stationarity is essential for many time series models.

## Seasonal Decomposition
Seasonal decomposition involves breaking down a time series into its components:
- **Trend**
- **Seasonal**
- **Residual**

### Example: Airline Passengers Dataset
Using the `statsmodels` library, we can decompose the airline passengers dataset to observe these components.

```python
import pandas as pd
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv('airline_passengers.csv', index_col='Month', parse_dates=True)

# Perform seasonal decomposition
decomposition = sm.tsa.seasonal_decompose(data['#Passengers'], model='multiplicative')
decomposition.plot()
```

## Transformations for Stationarity
To achieve stationarity, we can apply transformations such as:
- **Differencing**: Subtracting the previous observation from the current observation.
- **Logarithms**: Applying a logarithmic transformation to stabilize variance.

### Example: Differencing
```python
data['#Passengers_diff'] = data['#Passengers'].diff()
```

## Autocorrelation and Partial Autocorrelation
### Autocorrelation Function (ACF)
ACF measures the correlation between observations at different lags.

### Partial Autocorrelation Function (PACF)
PACF measures the correlation between observations at different lags, controlling for the correlations at shorter lags.

## References
- Brockwell, P. J., & Davis, R. A. (2002). Introduction to Time Series and Forecasting.
- Shumway, R. H., & Stoffer, D. S. (2017). Time Series Analysis and Its Applications.
- Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice.