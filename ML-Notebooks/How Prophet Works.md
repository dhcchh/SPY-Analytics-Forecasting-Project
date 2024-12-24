# Why Log Returns are Not Feasible for Price Prediction with Prophet

Log returns are widely used in financial analysis due to their additive nature and ability to normalize price changes. However, their use in long-term forecasting with Prophet poses significant challenges. Prophet relies on historical patterns to predict future values, assuming trends and seasonality observed in the past will persist.

---

## How Prophet Works

Prophet decomposes time series data into the following components:

\[
y(t) = g(t) + s(t) + h(t) + \epsilon_t
\]

Where:
1. **`g(t)` (Trend)**: Captures long-term growth patterns.
2. **`s(t)` (Seasonality)**: Models periodic changes, such as yearly or weekly cycles.
3. **`h(t)` (Holidays/Events)**: Adds custom impacts for specific dates.
4. **`ε_t` (Noise)**: Accounts for random fluctuations.

### Trend Component (`g(t)`)
The trend is modeled as a piecewise linear or logistic growth function:
\[
g(t) = k + mt + \sum_{i} a_i \delta_i(t)
\]
- \( k \): Offset (starting value of the trend).
- \( m \): Growth rate.
- \( a_i \): Change in growth rate at specific points.
- \( \delta_i(t) \): Indicator for change points.

### Seasonality Component (`s(t)`)
Seasonality is modeled with a Fourier series:
\[
s(t) = \sum_{n=1}^N \left( a_n \cos\left(\frac{2 \pi n t}{T}\right) + b_n \sin\left(\frac{2 \pi n t}{T}\right) \right)
\]
- \( N \): Number of Fourier terms (controls complexity of cycles).
- \( T \): Period of the cycle (e.g., 365.25 for yearly seasonality).

### Uncertainty Intervals
Prophet estimates uncertainty intervals using Monte Carlo sampling, which reflects the range of likely future outcomes.

---

## Why Log Returns are Problematic with Prophet

### 1. Exponential Price Conversion
Log returns are additive, but converting them back to prices involves compounding:
\[
P_t = P_0 \times e^{\text{cumulative log return}}
\]
Even small forecast errors in log returns (\( y(t) \)) lead to exponential errors in prices over long horizons.

### 2. Trend Extrapolation
Prophet models the trend component \( g(t) \) assuming historical patterns will persist. If past log returns exhibit a strong upward trend (e.g., due to a prolonged bull market), Prophet extrapolates this, resulting in exaggerated log return predictions. When converted to prices, this leads to unrealistic price inflation.

### 3. Seasonality Mismatch
Seasonal patterns in log returns (e.g., monthly or yearly cycles) may not translate directly to price movements. Prophet models log returns additively, but price changes are inherently multiplicative, leading to inconsistencies in the forecast.

---

## Example: Exponential Compounding in Price Prediction

### Forecasting Log Returns
Suppose Prophet predicts daily log returns as:
\[
y(t) = 0.001 + \epsilon_t
\]
Where \( \epsilon_t \) represents random noise.

### Convert to Prices
Using the cumulative log return:
\[
P_t = P_0 \times e^{\sum y(t)}
\]
- Assume \( P_0 = 100 \) (last known price).

#### After 30 Days:
\[
\text{Cumulative log return} = 0.001 \times 30 = 0.03
\]
\[
P_t = 100 \times e^{0.03} \approx 103.05
\]

#### After 365 Days:
\[
\text{Cumulative log return} = 0.001 \times 365 = 0.365
\]
\[
P_t = 100 \times e^{0.365} \approx 144.33
\]

### Impact of Small Errors
If the model slightly overestimates log returns (\( y(t) = 0.002 \)):

#### After 365 Days:
\[
\text{Cumulative log return} = 0.002 \times 365 = 0.73
\]
\[
P_t = 100 \times e^{0.73} \approx 208.29
\]

This compounding effect demonstrates how small errors in log returns lead to exponentially inflated price predictions.

---

## Performance Metrics

Prophet evaluates performance using metrics like `mse`, `rmse`, `mae`, etc. These metrics measure the error in log return predictions, but they do not account for the exponential growth in price errors when converting log returns back to prices.

For example:
- An RMSE of 0.01 in log returns may seem acceptable, but over a long horizon, this translates to significant overestimation in price predictions due to compounding.

---

## Conclusion

Using log returns with Prophet for price prediction is not feasible due to:
1. **Exponential compounding of small errors** when converting back to prices.
2. **Trend extrapolation bias**, which amplifies long-term errors in log returns.
3. **Mismatch between additive modeling of log returns** and the multiplicative nature of price changes.

Using price data directly allows Prophet to model trends and seasonality in a way that better reflects market dynamics, avoiding distortions introduced by cumulative log returns.
