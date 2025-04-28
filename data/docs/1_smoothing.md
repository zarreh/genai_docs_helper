**Smoothing Methods**

Smoothing methods are techniques used to remove noise from data and reveal underlying patterns. They are essential in time series analysis, signal processing, and various other fields. Here are some common smoothing methods:

1. **Moving Average**:
   - **Simple Moving Average (SMA)**: Calculates the average of a fixed number of past observations.
   - **Weighted Moving Average (WMA)**: Assigns different weights to past observations, giving more importance to recent data.
   - **Exponential Moving Average (EMA)**: Applies exponentially decreasing weights to past observations, making it more responsive to recent changes.

2. **Kernel Smoothing**:
   - **Gaussian Kernel**: Uses a Gaussian function to weigh observations, providing smooth estimates.
   - **Epanechnikov Kernel**: Applies a quadratic function, often used for its efficiency in computation.

3. **Spline Smoothing**:
   - **Cubic Splines**: Fits piecewise cubic polynomials to the data, ensuring smooth transitions at the knots.
   - **B-Splines**: Utilizes basis splines for flexible and efficient smoothing.

4. **Local Regression (LOESS)**:
   - **LOESS**: Fits multiple local regressions to subsets of the data, providing a smooth curve that adapts to local variations.

5. **Savitzky-Golay Filter**:
   - **Savitzky-Golay**: Applies polynomial fitting to a moving window, preserving higher moments and reducing noise.

6. **Fourier Transform**:
   - **Low-pass Filter**: Removes high-frequency noise by filtering out components above a certain threshold.

