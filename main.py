import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas_datareader.data as web
import pywt

# Data download & stock performance
tickers = ["GLD", "IEFA", "TLT", "SPY"]
end_date = date.today()
start_date = end_date - relativedelta(years=10)
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
adj_prices = data.xs(key="Close", level=0, axis=1)
rf_ticker = yf.Ticker("^IRX")
risk_free_rate = rf_ticker.history(period="1d")["Close"].iloc[-1] / 100
daily_returns = adj_prices.pct_change().dropna()

# Macro download and allignment
macro_data = web.DataReader(["CPIAUCSL", "INDPRO"], "fred", start_date, end_date)
macro_delayed = macro_data.shift(15, freq="D")
macro_table = pd.DataFrame(index=daily_returns.index)
macro_table = macro_table.join(macro_delayed, how="left").ffill()
aligned_prices = adj_prices.loc[daily_returns.index]
quantamental_table = pd.concat([aligned_prices, macro_table], axis=1)

# Wavelet trasnform (denoising macro's trend)
def wavelet_lowpass_filter(series, wavelet='db4', level=2):
    clean_series = series.dropna()
    if len(clean_series) < 10:
        return pd.Series(np.nan, index=series.index)
        
    clean_values = clean_series.values.copy()
        
    coefficients = pywt.wavedec(clean_values, wavelet, level=level)
    for i in range(1, len(coefficients)):
        coefficients[i] = np.zeros_like(coefficients[i])
        
    reconstructed_trend = pywt.waverec(coefficients, wavelet)
    result = pd.Series(reconstructed_trend[:len(clean_series)], index=clean_series.index)
    return result.reindex(series.index)

# SIGNAL EXTRACTION (RAW vs WAVELET vs MOVING AVERAGE)

# 3-month macro momentum (Raw Signal)
quantamental_table['CPI_Momentum'] = quantamental_table['CPIAUCSL'].pct_change(63)
quantamental_table['Growth_Momentum'] = quantamental_table['INDPRO'].pct_change(63)

# Filtered Signal via Wavelet
quantamental_table['CPI_Wavelet_Trend'] = wavelet_lowpass_filter(quantamental_table['CPI_Momentum'])
quantamental_table['Growth_Wavelet_Trend'] = wavelet_lowpass_filter(quantamental_table['Growth_Momentum'])

# Filtered Signal via Exponential Moving Average (EMA 3-Months / 63 Trading Days)
quantamental_table['CPI_EMA_Trend'] = quantamental_table['CPI_Momentum'].ewm(span=63, adjust=False).mean()
quantamental_table['Growth_EMA_Trend'] = quantamental_table['Growth_Momentum'].ewm(span=63, adjust=False).mean()

# Drop rows to align all active signals
quantamental_table = quantamental_table.dropna(subset=['CPI_Wavelet_Trend', 'Growth_Wavelet_Trend', 'CPI_EMA_Trend', 'Growth_EMA_Trend']).copy()

# Regime Mapping 
regime_names = ['Overheating', 'Goldilocks', 'Deflation', 'Stagflation']

# Wavelet Regimes
wavelet_conditions = [
    (quantamental_table['Growth_Wavelet_Trend'] >= 0) & (quantamental_table['CPI_Wavelet_Trend'] >= 0),
    (quantamental_table['Growth_Wavelet_Trend'] >= 0) & (quantamental_table['CPI_Wavelet_Trend'] < 0),
    (quantamental_table['Growth_Wavelet_Trend'] < 0) & (quantamental_table['CPI_Wavelet_Trend'] < 0),
    (quantamental_table['Growth_Wavelet_Trend'] < 0) & (quantamental_table['CPI_Wavelet_Trend'] >= 0)
]
quantamental_table['Wavelet_Regime'] = np.select(wavelet_conditions, regime_names, default='Unknown')

# Raw (No Filter) Regimes
raw_conditions = [
    (quantamental_table['Growth_Momentum'] >= 0) & (quantamental_table['CPI_Momentum'] >= 0),
    (quantamental_table['Growth_Momentum'] >= 0) & (quantamental_table['CPI_Momentum'] < 0),
    (quantamental_table['Growth_Momentum'] < 0) & (quantamental_table['CPI_Momentum'] < 0),
    (quantamental_table['Growth_Momentum'] < 0) & (quantamental_table['CPI_Momentum'] >= 0)
]
quantamental_table['Raw_Regime'] = np.select(raw_conditions, regime_names, default='Unknown')

# EMA (Moving Average Filter) Regimes
ema_conditions = [
    (quantamental_table['Growth_EMA_Trend'] >= 0) & (quantamental_table['CPI_EMA_Trend'] >= 0),
    (quantamental_table['Growth_EMA_Trend'] >= 0) & (quantamental_table['CPI_EMA_Trend'] < 0),
    (quantamental_table['Growth_EMA_Trend'] < 0) & (quantamental_table['CPI_EMA_Trend'] < 0),
    (quantamental_table['Growth_EMA_Trend'] < 0) & (quantamental_table['CPI_EMA_Trend'] >= 0)
]
quantamental_table['EMA_Regime'] = np.select(ema_conditions, regime_names, default='Unknown')

# Backtest & dynamic weights allocation
weight_matrix = {
    'Goldilocks':       [0.05, 0.35, 0.10, 0.50],
    'Overheating':      [0.25, 0.20, 0.05, 0.50],
    'Stagflation':     [0.60, 0.10, 0.10, 0.20],
    'Deflation':       [0.05, 0.10, 0.60, 0.25]
}

# Shift weights by 1 day to strictly avoid look-ahead bias
wavelet_weights = pd.DataFrame([weight_matrix[r] for r in quantamental_table['Wavelet_Regime']], index=quantamental_table.index, columns=tickers).shift(1).dropna()
raw_weights = pd.DataFrame([weight_matrix[r] for r in quantamental_table['Raw_Regime']], index=quantamental_table.index, columns=tickers).shift(1).dropna()
ema_weights = pd.DataFrame([weight_matrix[r] for r in quantamental_table['EMA_Regime']], index=quantamental_table.index, columns=tickers).shift(1).dropna()

# Find intersection index to run a fair backtest 
common_index = wavelet_weights.index.intersection(raw_weights.index).intersection(ema_weights.index)
filtered_returns = daily_returns.loc[common_index]
wavelet_weights = wavelet_weights.loc[common_index]
raw_weights = raw_weights.loc[common_index]
ema_weights = ema_weights.loc[common_index]

# Performances
wavelet_strat_returns = (filtered_returns * wavelet_weights).sum(axis=1)
cum_wavelet_returns = (1 + wavelet_strat_returns).cumprod() - 1
raw_strat_returns = (filtered_returns * raw_weights).sum(axis=1)
cum_raw_returns = (1 + raw_strat_returns).cumprod() - 1
ema_strat_returns = (filtered_returns * ema_weights).sum(axis=1)
cum_ema_returns = (1 + ema_strat_returns).cumprod() - 1

# Annualized Turnover Metrics
wavelet_turnover = wavelet_weights.diff().abs().sum(axis=1).mean() * 252
raw_turnover = raw_weights.diff().abs().sum(axis=1).mean() * 252
ema_turnover = ema_weights.diff().abs().sum(axis=1).mean() * 252

# Performances & Plot
# ==============================================================================
def calculate_max_drawdown(cum_returns_series):
    equity_curve = cum_returns_series + 1
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()

print("--- PURE REACTIVITY REGIME SWITCHES PROFILE (NO THRESHOLD) ---")
print(f"Wavelet Strategy Total Regime Changes:       {quantamental_table['Wavelet_Regime'].ne(quantamental_table['Wavelet_Regime'].shift()).sum()}")
print(f"EMA Moving Average Strategy Total Changes:   {quantamental_table['EMA_Regime'].ne(quantamental_table['EMA_Regime'].shift()).sum()}")
print(f"Raw (No Filter) Strategy Total Changes:      {quantamental_table['Raw_Regime'].ne(quantamental_table['Raw_Regime'].shift()).sum()}")
print("-" * 50)

# Performance calculations
w_ann_return = wavelet_strat_returns.mean() * 252
w_ann_vol = wavelet_strat_returns.std() * np.sqrt(252)
w_sharpe = (w_ann_return - risk_free_rate) / w_ann_vol
w_max_dd = calculate_max_drawdown(cum_wavelet_returns)

r_ann_return = raw_strat_returns.mean() * 252
r_ann_vol = raw_strat_returns.std() * np.sqrt(252)
r_sharpe = (r_ann_return - risk_free_rate) / r_ann_vol
r_max_dd = calculate_max_drawdown(cum_raw_returns)

e_ann_return = ema_strat_returns.mean() * 252
e_ann_vol = ema_strat_returns.std() * np.sqrt(252)
e_sharpe = (e_ann_return - risk_free_rate) / e_ann_vol
e_max_dd = calculate_max_drawdown(cum_ema_returns)

print(f"1. Quantamental (Wavelet) -> Return: {w_ann_return:.2%}, Volatility: {w_ann_vol:.2%}, Sharpe: {w_sharpe:.2f}, MaxDD: {w_max_dd:.2%}, Turnover: {wavelet_turnover:.2%}")
print(f"2. Quantamental (EMA 3M)  -> Return: {e_ann_return:.2%}, Volatility: {e_ann_vol:.2%}, Sharpe: {e_sharpe:.2f}, MaxDD: {e_max_dd:.2%}, Turnover: {ema_turnover:.2%}")
print(f"3. Quantamental (Raw)     -> Return: {r_ann_return:.2%}, Volatility: {r_ann_vol:.2%}, Sharpe: {r_sharpe:.2f}, MaxDD: {r_max_dd:.2%}, Turnover: {raw_turnover:.2%}")

plt.figure(figsize=(12, 6))
plt.plot(cum_wavelet_returns * 100, label='Quantamental Strategy (Wavelet Filter)', color='darkblue', lw=2)
plt.plot(cum_ema_returns * 100, label='Quantamental Strategy (EMA 3M Filter)', color='forestgreen', lw=1.5)
plt.plot(cum_raw_returns * 100, label='Quantamental Strategy (Raw - Pure Reactivity)', color='crimson', lw=1.2, linestyle='--')
plt.title('10-Year Clean Filter Battle (Pure Reactivity Framework)')
plt.ylabel('Cumulative Return (%)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('performance_chart.png', dpi=300, bbox_inches='tight') 
plt.show()
