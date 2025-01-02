# Investment Analytics: Highlighting the Value of Stock Market Investments for Long-Term Growth

## Project Directory Structure
- **SPY-Analytics-Forecasting-Project/**
  - **Data/**
    - `cpi.csv`
    - `spy_full.csv`
    - `spy_month.csv`
    - `spy_week.csv`
  - **Forecasts/**
    - `lstm_forecast.csv`
    - `montecarlo_forecast.csv`
    - `prophet_forecast.csv`
  - **ML-Notebooks/**
    - `Final-Calculations.ipynb`
    - `LSTM_helper.py`
    - `ML-ARIMA-forecast.ipynb`
    - `ML-LSTM.ipynb`
    - `ML-Prophet-LogReturn.ipynb`
    - `ML-Prophet-Price.ipynb`
    - `ML-XGB-FODP.ipynb`
    - `ML-XGB-LR.ipynb`
    - `Monte-Carlo.ipynb`
    - `Regression-Baseline.ipynb`
  - **Notebooks-Analysis/**
    - `analysis_nominal.ipynb`
    - `analysis_real.ipynb`
  - `.gitignore`
  - `README.md`

### **Introduction**

This project explores the power and potential of investing in the S&P 500 ETF (SPY), one of the most widely held and traded ETFs that tracks the performance of the S&P 500 index. By analyzing historical returns, evaluating its effectiveness as a hedge against inflation, and applying machine learning models to predict its future price, the project offers a comprehensive view of SPY's investment potential in different market conditions.

The project is structured into three distinct parts:

1. **Historical Returns Analysis**: 
   We dive into the historical performance of SPY, showcasing the power of long-term investing by analyzing its returns over various time periods. This section highlights the benefits of holding SPY as a foundational asset for portfolio growth.

2. **Hedge Against Inflation**: 
   In this section, we evaluate SPY's ability to act as a hedge against inflation. With rising concerns about inflation in today's economic landscape, we explore how SPY performs during inflationary periods and whether it provides protection for investors' purchasing power.

3. **Price Prediction Using Machine Learning**: 
   The final part of the project leverages machine learning models to forecast SPY's future price. This section combines time-series modeling with advanced machine learning techniques to offer insights into the future trajectory of SPY.

This repository contains the code and models used in the project to demonstrate the power of investing in SPY, and its potential future outlook.

If you are interested in reading more about the project, [I published a blog on my website](https://www.notion.so/chan-dinghao/Investment-Analytics-Highlighting-the-Value-of-Stock-Market-Investments-for-Long-Term-Growth-7e6f10bafb9640c9bb8706fe32cd0ba0) that goes into deep detail about the inspiration anad analysis for the project.

**Table of contents**
- Part 1.1: Analysing the performance of an index fund like SPY compared to a traditional savings account.
- Part 1.2: Analysing the power of ETFs like SPY as a hedge against inflation.
- Part 2 : Predicting the returns of SPY in the future using various ML and Time-Series methods.

# Insights Derived. 
Key Metrics Considered (Currencies are in USD) :
- Nominal ROI. How much your investment in SPY yielded based on your initial investments.
- Real ROI. How much your investment in SPY yielded based on your initial investments, adjusted for inflation.
## Part 1.1 
[Part 1 Analysis](https://chan-dinghao.notion.site/Part-1-1-Yielding-Significant-Returns-10120b01262d4acf88637f0bc6bd3c03?pvs=4)
- Consider only nominal ROI.  
- You will not save enough to comfortably retire solely relying on interest rates from saving accounts.
- The investor does not need any investing knowledge, they just need to open a brokerage account and purchase SPY.
- The investor must be patient. They may not see returns in the short runs, but they will definitely see  returns in the long run.
- In my opinion, the investor must contribute US$1000 monthly in order to see significant returns.
## Part 1.2
[Part 1.2 Analysis](https://chan-dinghao.notion.site/Part-1-2-Hedging-Against-Inflation-624736b7e8c6406aa239529ce2ed8d4c?pvs=74)
- You will lose about 40-45% of your purchasing power when not accounting for inflation saving money in a low yield savings account.
- Investing a significant amount in SPY can still allow your net worth to grow despite inflation. The growth rate of SPY effectively outpaces the inflation rate.
  
## Part 2: Future Returns Forecasting
[Part 2 Forecasting](https://chan-dinghao.notion.site/Part-2-Predicting-Future-Returns-14528ad3155f800db3b5e9f9739176a4?pvs=4)

For those who have not yet started investing, the question often arises: *Is it too late to begin?* In this section, we aim to address this by leveraging machine learning to forecast the potential future prices of SPY over the next 20 years. By analyzing historical trends and incorporating advanced predictive models, we provide insights into the long-term growth prospects of SPY, offering a data-driven perspective on why starting today could still lead to significant financial growth in the future.

### Key Highlights

1. **Forecasting Models**:
   - **Monte Carlo Simulation**: Captures a range of potential outcomes based on historical data and random sampling.
   - **Prophet**: Utilized for time series forecasting to predict SPY prices within confidence intervals.
   - **LSTM (Long Short-Term Memory)**: A deep learning model trained to predict future SPY prices based on past trends.

2. **Final Results**:
   - We take the average of all mean predictions, resulting in a forecasted price of SPY in 2044 at **US$ 2507.39**.
   - This suggests SPY could grow approximately **5 times** in the next 20 years.

3. **Benchmarks**:
   - The forecasted value aligns with polynomial growth models, which predict SPY prices ranging from **US$ 1747** to **US$ 5368**.
   - While the prediction falls short of a hypothetical 6x growth rate, it is consistent with the historical annualized return of SPY (~10% yearly).

4. **Investor Takeaway**:
   - Predicting future SPY prices serves as a **gauge of potential trends**, not a definitive forecast.
   - Despite uncertainties, the analysis reinforces SPY's long-term growth potential, making it an attractive option for patient investors.

### Important Note

Predicting financial markets is inherently uncertain. Factors such as macroeconomic conditions, geopolitical events, and investor sentiment, which are difficult to capture in historical data, play a significant role in shaping market outcomes. As such, these forecasts should be interpreted as tools for insight rather than guarantees of future performance.

If you are only interested in the results, feel free to skip directly to the **Final Results Section** above for a summary of key findings.
