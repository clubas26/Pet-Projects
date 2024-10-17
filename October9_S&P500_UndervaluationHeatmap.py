import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warningspyth
import warnings
warnings.filterwarnings('ignore')

# Load the list of S&P 500 companies
sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
sp500_symbols = sp500_table[0]['Symbol'].tolist()

# Initialize a list to store financial metrics
metrics = []

print("Retrieving data for S&P 500 companies...")

# Iterate over the symbols and retrieve financial data
for symbol in sp500_symbols:
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        pe_ratio = info.get('trailingPE', np.nan)
        pb_ratio = info.get('priceToBook', np.nan)
        peg_ratio = info.get('pegRatio', np.nan)
        dividend_yield = info.get('dividendYield', np.nan)
        debt_to_equity = info.get('debtToEquity', np.nan)
        roe = info.get('returnOnEquity', np.nan)
        free_cash_flow = info.get('freeCashflow', np.nan)
        market_cap = info.get('marketCap', np.nan)

        # Calculate Free Cash Flow Yield
        if free_cash_flow and market_cap and market_cap != 0:
            fcf_yield = free_cash_flow / market_cap
        else:
            fcf_yield = np.nan

        metrics.append({
            'Symbol': symbol,
            'P/E Ratio': pe_ratio,
            'P/B Ratio': pb_ratio,
            'PEG Ratio': peg_ratio,
            'Dividend Yield': dividend_yield,
            'Debt-to-Equity': debt_to_equity,
            'ROE': roe,
            'FCF Yield': fcf_yield
        })
    except Exception as e:
        print(f"Failed to get data for {symbol}: {e}")

# Create a DataFrame
df = pd.DataFrame(metrics)
df.set_index('Symbol', inplace=True)

# Drop rows with missing data
df.dropna(inplace=True)

# Standardize the metrics
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Invert the metrics where lower is better
for col in ['P/E Ratio', 'P/B Ratio', 'PEG Ratio', 'Debt-to-Equity']:
    df_scaled[col] = -df_scaled[col]

# Assign weights
weights = {
    'P/E Ratio': 0.25,
    'P/B Ratio': 0.15,
    'PEG Ratio': 0.20,
    'Dividend Yield': 0.10,
    'Debt-to-Equity': 0.10,
    'ROE': 0.10,
    'FCF Yield': 0.10
}

# Calculate the undervaluation score
df_scaled['Undervaluation Score'] = df_scaled.apply(
    lambda row: sum(row[col] * weight for col, weight in weights.items()), axis=1
)

# Get the top 10 most undervalued stocks
top10 = df_scaled.sort_values('Undervaluation Score', ascending=False).head(10)

# Prepare data for heatmap
heatmap_data = df.loc[top10.index]

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Top 10 Most Undervalued S&P 500 Stocks')
plt.ylabel('Stock Symbol')
plt.tight_layout()
plt.show()