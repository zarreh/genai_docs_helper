# Time Series Feature Engineering and Selection Strategy

## Document Information
- **Document Title**: Time Series Feature Engineering and Selection Strategy
- **Version**: 1.4
- **Last Updated**: July 2025
- **Author**: Data Science Team
- **Reviewers**: ML Engineering Team, Business Analytics
- **Related Documents**: Retail Demand Forecasting System Architecture v1.2

---

## 1. Executive Summary

This document outlines the comprehensive feature engineering and selection methodology employed in the Retail Demand Forecasting System. Our approach combines domain expertise with automated feature selection techniques to create robust predictive features that capture temporal patterns, seasonal effects, and external influences on warehouse order volumes.

### Key Achievements
- **Feature Performance**: 127 engineered features with 89% predictive power retention
- **Model Improvement**: 23% MAPE reduction compared to baseline raw features
- **Processing Efficiency**: Feature pipeline processes 2.1M records in <12 minutes
- **Warehouse Coverage**: Standardized features across 50+ warehouse locations

---

## 2. Feature Engineering Philosophy

### 2.1 Design Principles

1. **Temporal Awareness**: Capture time-dependent patterns at multiple scales
2. **Business Context**: Incorporate domain knowledge about retail operations
3. **Scalability**: Ensure features work across diverse warehouse characteristics
4. **Interpretability**: Maintain explainable features for business stakeholders
5. **Robustness**: Handle missing data and outliers gracefully

### 2.2 Feature Categories

```
Feature Taxonomy:
├── Temporal Features (32 features)
│   ├── Lag Features (12)
│   ├── Rolling Statistics (15)
│   └── Seasonal Components (5)
├── Calendar Features (18 features)
│   ├── Date Components (8)
│   ├── Holiday Features (6)
│   └── Business Calendar (4)
├── External Features (24 features)
│   ├── Weather Features (12)
│   ├── Economic Indicators (6)
│   └── Operational Events (6)
├── Warehouse-Specific Features (31 features)
│   ├── Historical Patterns (15)
│   ├── Capacity Metrics (8)
│   └── Performance Indicators (8)
└── Interaction Features (22 features)
    ├── Cross-Feature Products (12)
    └── Conditional Features (10)
```

---

## 3. Temporal Feature Engineering

### 3.1 Lag Features

**Purpose**: Capture autoregressive patterns and dependencies

**Implementation Strategy**:
```python
# Lag feature configuration
LAG_PERIODS = {
    'short_term': [1, 2, 3, 7],      # Recent trends
    'medium_term': [14, 21, 28],     # Monthly patterns
    'long_term': [90, 180, 365]      # Seasonal patterns
}

# Example lag feature creation
def create_lag_features(df, target_col='order_volume', lags=LAG_PERIODS):
    features = df.copy()

    for period_type, periods in lags.items():
        for lag in periods:
            col_name = f'{target_col}_lag_{lag}d'
            features[col_name] = features.groupby('warehouse_id')[target_col].shift(lag)

            # Add lag interaction with day of week
            features[f'{col_name}_dow_interaction'] = (
                features[col_name] * features['day_of_week']
            )

    return features
```

**Performance Analysis**:
| Lag Period | Feature Importance | Business Interpretation |
|------------|-------------------|------------------------|
| 1-day | 0.18 | Yesterday's momentum |
| 7-day | 0.24 | Weekly seasonality |
| 14-day | 0.15 | Bi-weekly patterns |
| 30-day | 0.12 | Monthly cycles |
| 365-day | 0.21 | Yearly seasonality |

### 3.2 Rolling Statistics

**Purpose**: Smooth short-term noise and capture trend information

**Window Configurations**:
```python
ROLLING_WINDOWS = {
    'trend_windows': [7, 14, 30],        # Trend detection
    'volatility_windows': [7, 21],       # Volatility measures
    'seasonal_windows': [28, 91, 182]    # Seasonal smoothing
}

ROLLING_FUNCTIONS = {
    'central_tendency': ['mean', 'median'],
    'dispersion': ['std', 'var', 'iqr'],
    'extremes': ['min', 'max', 'quantile_25', 'quantile_75'],
    'shape': ['skew', 'kurtosis']
}
```

**Feature Examples**:
- `order_volume_rolling_7d_mean`: 7-day moving average
- `order_volume_rolling_30d_std`: 30-day volatility
- `order_volume_rolling_14d_trend`: Linear trend over 14 days
- `order_volume_rolling_7d_cv`: Coefficient of variation

**Statistical Validation**:
- **Stationarity Test**: Augmented Dickey-Fuller p-value < 0.05 for 94% of features
- **Autocorrelation**: Significant ACF values up to lag 30 for most warehouses
- **Cross-correlation**: Weather features show 2-3 day lag correlation (r=0.31)

### 3.3 Seasonal Decomposition

**Purpose**: Extract trend, seasonal, and residual components

**Decomposition Method**: STL (Seasonal and Trend decomposition using Loess)

```python
# Seasonal decomposition configuration
SEASONAL_CONFIG = {
    'daily_seasonality': {'period': 7, 'seasonal': 7},
    'monthly_seasonality': {'period': 30.44, 'seasonal': 13},
    'yearly_seasonality': {'period': 365.25, 'seasonal': 25}
}

# Generated seasonal features
seasonal_features = [
    'trend_component',           # Long-term trend
    'weekly_seasonal',          # Day-of-week effect
    'monthly_seasonal',         # Day-of-month effect
    'yearly_seasonal',          # Day-of-year effect
    'residual_component'        # Unexplained variance
]
```

**Seasonal Pattern Analysis**:
- **Weekly Seasonality**: Peak on Tuesdays (1.34x average), low on Sundays (0.67x)
- **Monthly Seasonality**: Higher volumes in first half of month (1.12x average)
- **Yearly Seasonality**: Q4 peak (1.28x), Q1 trough (0.89x)

---

## 4. Calendar and Holiday Features

### 4.1 Date Component Features

**Basic Date Features**:
```python
DATE_FEATURES = {
    'temporal_components': [
        'year', 'month', 'day', 'day_of_week', 'day_of_year',
        'week_of_year', 'quarter', 'is_weekend'
    ],
    'cyclical_encoding': [
        'month_sin', 'month_cos',           # Monthly cycles
        'day_of_week_sin', 'day_of_week_cos', # Weekly cycles
        'day_of_year_sin', 'day_of_year_cos'  # Yearly cycles
    ]
}
```

**Advanced Calendar Features**:
- `days_since_month_start`: Position within month
- `days_until_month_end`: Distance to month end
- `is_month_end`: Last 3 days of month (payroll effect)
- `is_quarter_end`: Business quarter boundaries
- `working_days_in_month`: Business day count

### 4.2 Holiday Feature Engineering

**Holiday Categories**:
```python
HOLIDAY_TYPES = {
    'national_holidays': [
        'New Year', 'Easter Monday', 'Labour Day', 
        'Liberation Day', 'Christmas Day', 'Boxing Day'
    ],
    'regional_holidays': [
        'St. Nicholas Day', 'Czech Statehood Day'
    ],
    'commercial_events': [
        'Black Friday', 'Cyber Monday', 'Valentine Day'
    ],
    'seasonal_periods': [
        'Summer Holidays', 'Winter Holidays', 'Spring Break'
    ]
}
```

**Holiday Impact Features**:
- `is_holiday`: Binary holiday indicator
- `holiday_type`: Categorical holiday classification
- `days_to_holiday`: Distance to next holiday (1-30 days)
- `days_from_holiday`: Distance from last holiday (1-30 days)
- `holiday_cluster`: Multi-day holiday periods
- `pre_holiday_effect`: 1-3 days before major holidays
- `post_holiday_effect`: 1-2 days after major holidays

**Holiday Impact Analysis**:
| Holiday Type | Pre-Holiday Effect | Holiday Effect | Post-Holiday Effect |
|--------------|-------------------|----------------|-------------------|
| Christmas | +45% (2 days before) | -78% | -23% (1 day after) |
| Easter | +12% (1 day before) | -45% | +8% (1 day after) |
| Black Friday | +67% (1 day before) | +156% | +23% (2 days after) |
| National Holidays | +8% (1 day before) | -34% | -5% (1 day after) |

---

## 5. External Factor Features

### 5.1 Weather Features

**Raw Weather Variables**:
- Temperature (min, max, average)
- Precipitation (amount, probability)
- Wind speed and direction
- Humidity and pressure
- Cloud cover and visibility

**Engineered Weather Features**:
```python
WEATHER_FEATURES = {
    'comfort_indices': [
        'heat_index',           # Perceived temperature
        'wind_chill',          # Cold weather comfort
        'comfort_score'        # Overall weather comfort
    ],
    'extreme_conditions': [
        'is_extreme_cold',     # < -10°C
        'is_extreme_hot',      # > 35°C
        'is_heavy_rain',       # > 20mm/day
        'is_stormy'           # Wind > 50 km/h
    ],
    'weather_trends': [
        'temp_3d_trend',       # 3-day temperature trend
        'precip_7d_sum',       # Weekly precipitation
        'weather_volatility'   # Weather stability index
    ]
}
```

**Weather Impact Correlation**:
- **Temperature vs Orders**: r = -0.23 (higher temp → fewer orders)
- **Precipitation vs Orders**: r = +0.18 (rain → more orders)
- **Extreme Weather**: 15-25% order volume deviation
- **Seasonal Weather Adjustment**: Improves MAPE by 2.1%

### 5.2 Economic and Market Features

**Economic Indicators**:
- Consumer Price Index (CPI)
- Unemployment rate
- GDP growth rate
- Consumer confidence index
- Retail sales index

**Market Features**:
- Competitor promotion periods
- Market share indicators
- Industry seasonal adjustments
- Economic calendar events

---

## 6. Warehouse-Specific Features

### 6.1 Operational Characteristics

**Capacity and Performance Metrics**:
```python
WAREHOUSE_FEATURES = {
    'capacity_metrics': [
        'max_daily_capacity',      # Maximum processing capacity
        'avg_utilization_rate',    # Historical utilization
        'capacity_utilization',    # Current vs max capacity
        'bottleneck_score'         # Operational constraints
    ],
    'performance_indicators': [
        'avg_processing_time',     # Order processing speed
        'error_rate',             # Operational error rate
        'staff_efficiency',       # Productivity metrics
        'equipment_uptime'        # Operational reliability
    ],
    'historical_patterns': [
        'warehouse_seasonality',   # Location-specific patterns
        'growth_trend',           # Long-term growth rate
        'volatility_score',       # Order volume stability
        'peak_day_pattern'        # Busiest day patterns
    ]
}
```

### 6.2 Location-Based Features

**Geographic Features**:
- Warehouse coordinates (lat, lon)
- Population density in catchment area
- Distance to major cities
- Transportation hub proximity
- Regional economic indicators

**Demographic Features**:
- Age distribution in service area
- Income levels
- Urban vs rural classification
- Shopping behavior patterns

---

## 7. Feature Selection Methodology

### 7.1 Selection Pipeline

```python
FEATURE_SELECTION_PIPELINE = {
    'stage_1_filtering': {
        'method': 'variance_threshold',
        'threshold': 0.01,
        'features_removed': 23
    },
    'stage_2_correlation': {
        'method': 'correlation_matrix',
        'threshold': 0.95,
        'features_removed': 18
    },
    'stage_3_univariate': {
        'method': 'mutual_info_regression',
        'top_k': 100,
        'features_selected': 100
    },
    'stage_4_multivariate': {
        'method': 'recursive_feature_elimination',
        'estimator': 'random_forest',
        'features_selected': 67
    },
    'stage_5_stability': {
        'method': 'stability_selection',
        'bootstrap_samples': 100,
        'features_selected': 45
    }
}
```

### 7.2 Feature Importance Analysis

**Top 15 Most Important Features**:
| Rank | Feature Name | Importance | Category |
|------|-------------|------------|----------|
| 1 | order_volume_lag_7d | 0.142 | Temporal |
| 2 | yearly_seasonal | 0.128 | Seasonal |
| 3 | is_holiday | 0.089 | Calendar |
| 4 | order_volume_rolling_30d_mean | 0.076 | Temporal |
| 5 | day_of_week | 0.071 | Calendar |
| 6 | temperature_avg | 0.063 | Weather |
| 7 | days_to_holiday | 0.058 | Calendar |
| 8 | warehouse_capacity_utilization | 0.055 | Warehouse |
| 9 | order_volume_lag_1d | 0.052 | Temporal |
| 10 | monthly_seasonal | 0.048 | Seasonal |
| 11 | precipitation_amount | 0.045 | Weather |
| 12 | trend_component | 0.043 | Seasonal |
| 13 | is_weekend | 0.041 | Calendar |
| 14 | order_volume_rolling_7d_std | 0.039 | Temporal |
| 15 | quarter | 0.037 | Calendar |

### 7.3 Feature Stability Assessment

**Stability Metrics**:
- **Selection Frequency**: 89% of features selected in >80% of bootstrap samples
- **Importance Variance**: CV = 0.23 across different time periods
- **Cross-Warehouse Consistency**: 76% of features important across all warehouses
- **Temporal Stability**: 91% feature importance correlation between quarters

---

## 8. Feature Engineering Pipeline

### 8.1 Production Pipeline Architecture

```python
class FeatureEngineeringPipeline:
    def __init__(self):
        self.transformers = {
            'temporal': TemporalFeatureTransformer(),
            'calendar': CalendarFeatureTransformer(),
            'weather': WeatherFeatureTransformer(),
            'warehouse': WarehouseFeatureTransformer(),
            'interactions': InteractionFeatureTransformer()
        }
        self.selector = FeatureSelector()
        self.scaler = StandardScaler()

    def fit_transform(self, X, y=None):
        # Apply feature transformers
        features = []
        for name, transformer in self.transformers.items():
            feature_subset = transformer.fit_transform(X)
            features.append(feature_subset)

        # Combine all features
        X_engineered = pd.concat(features, axis=1)

        # Feature selection
        X_selected = self.selector.fit_transform(X_engineered, y)

        # Scaling
        X_scaled = self.scaler.fit_transform(X_selected)

        return X_scaled
```

### 8.2 Performance Metrics

**Pipeline Performance**:
- **Processing Time**: 11.3 minutes for full dataset (2.1M records)
- **Memory Usage**: Peak 8.2GB RAM during feature creation
- **Feature Creation Rate**: 185,000 records/minute
- **Error Rate**: <0.01% (robust error handling)

**Quality Metrics**:
- **Feature Completeness**: 99.7% (missing value handling)
- **Data Type Consistency**: 100% (automated type checking)
- **Range Validation**: 99.9% (outlier detection and capping)
- **Business Rule Compliance**: 100% (domain constraint validation)

---

## 9. Feature Monitoring and Maintenance

### 9.1 Drift Detection

**Feature Drift Monitoring**:
```python
DRIFT_DETECTION_CONFIG = {
    'statistical_tests': [
        'kolmogorov_smirnov',    # Distribution changes
        'chi_square',            # Categorical feature changes
        'population_stability_index'  # Overall stability
    ],
    'thresholds': {
        'warning': 0.1,          # Moderate drift
        'critical': 0.25         # Significant drift
    },
    'monitoring_frequency': 'daily',
    'lookback_window': 30        # Days for comparison
}
```

**Drift Alert Examples**:
- **Weather Feature Drift**: Seasonal temperature patterns shifted by 2 weeks (PSI = 0.15)
- **Holiday Pattern Change**: New regional holiday added (Chi-square p < 0.01)
- **Warehouse Capacity Update**: Capacity expansion affected utilization features

### 9.2 Feature Performance Tracking

**Performance Degradation Indicators**:
- Feature importance drop >20% over 30 days
- Correlation with target variable decrease >15%
- Prediction accuracy impact >2% MAPE increase
- Cross-validation stability score <0.8

### 9.3 Automated Feature Updates

**Update Triggers**:
- New data source integration
- Business rule changes
- Seasonal pattern shifts
- Model performance degradation

**Update Process**:
1. **Impact Assessment**: Analyze potential feature changes
2. **A/B Testing**: Compare old vs new features
3. **Gradual Rollout**: Phased deployment across warehouses
4. **Performance Monitoring**: Track impact on predictions
5. **Rollback Capability**: Revert if performance degrades

---

## 10. Best Practices and Guidelines

### 10.1 Feature Engineering Best Practices

1. **Domain Knowledge Integration**
   - Collaborate with business experts
   - Validate features against business logic
   - Document business interpretation

2. **Temporal Consistency**
   - Avoid data leakage from future information
   - Maintain consistent time zones
   - Handle daylight saving time transitions

3. **Scalability Considerations**
   - Optimize for computational efficiency
   - Design for new warehouse onboarding
   - Plan for increased data volume

4. **Interpretability Maintenance**
   - Keep feature names descriptive
   - Document feature creation logic
   - Provide business interpretation

### 10.2 Common Pitfalls and Solutions

| Pitfall | Impact | Solution |
|---------|--------|----------|
| Data Leakage | Overly optimistic performance | Strict temporal validation |
| Multicollinearity | Unstable model coefficients | Correlation analysis and VIF |
| Overfitting | Poor generalization | Cross-validation and regularization |
| Missing Value Propagation | Reduced data quality | Robust imputation strategies |
| Feature Explosion | Computational overhead | Systematic feature selection |

---

## 11. Future Enhancements

### 11.1 Short-term Improvements (Q3-Q4 2025)

- [ ] **Automated Feature Discovery**: ML-based feature generation
- [ ] **Real-time Feature Streaming**: Kafka-based feature updates
- [ ] **Advanced Interaction Detection**: Automated interaction mining
- [ ] **Feature Store Integration**: Centralized feature management

### 11.2 Long-term Vision (2026+)

- [ ] **Deep Feature Learning**: Neural network-based feature extraction
- [ ] **Causal Feature Engineering**: Causal inference-based features
- [ ] **Multi-modal Features**: Integration of image and text data
- [ ] **Federated Feature Learning**: Cross-warehouse feature sharing

---

## 12. Appendices

### 12.1 Feature Dictionary

**Complete feature catalog with 127 engineered features available in separate document**: `Feature_Dictionary_v1.4.xlsx`

### 12.2 Code Repository

**Feature engineering codebase**: `git@company.com:data-science/feature-engineering.git`

### 12.3 Performance Benchmarks

**Detailed performance analysis**: `Feature_Performance_Analysis_Q2_2025.pdf`

---

**Document Control**
- Next Review Date: October 2025
- Distribution: Data Science Team, ML Engineering, Business Analytics
- Classification: Internal Use Only
- Related Training Materials: Feature Engineering Workshop Series
