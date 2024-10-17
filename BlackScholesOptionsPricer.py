import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import warnings
import matplotlib

# Use a non-interactive backend for matplotlib
matplotlib.use('Agg')

# Suppress warnings
warnings.filterwarnings("ignore")

# Black-Scholes Option Pricing Model
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return price

# Streamlit App
def main():
    st.title("Options Pricing Application")
    st.sidebar.header("Option Parameters")
    
    # User Inputs
    S = st.sidebar.number_input("Stock Price (S)", min_value=1.0, value=100.0)
    K = st.sidebar.number_input("Strike Price (K)", min_value=1.0, value=100.0)
    T = st.sidebar.number_input("Time to Maturity (T) in Years", min_value=0.01, value=1.0)
    r = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=1.0, value=0.05)
    sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2)
    
    # Calculate Option Prices
    call_price = black_scholes(S, K, T, r, sigma, option_type="call")
    put_price = black_scholes(S, K, T, r, sigma, option_type="put")
    
    # Display Option Prices
    st.write(f"### Call Option Price: ${call_price:.2f}")
    st.write(f"### Put Option Price: ${put_price:.2f}")
    
    # Generate Heatmaps
    st.header("Heatmap of Option Prices")
    stock_prices = np.linspace(S * 0.5, S * 1.5, 50)
    volatilities = np.linspace(0.01, 1.0, 50)
    
    call_prices = np.zeros((len(stock_prices), len(volatilities)))
    put_prices = np.zeros((len(stock_prices), len(volatilities)))
    
    for i, s in enumerate(stock_prices):
        for j, vol in enumerate(volatilities):
            call_prices[i, j] = black_scholes(s, K, T, r, vol, option_type="call")
            put_prices[i, j] = black_scholes(s, K, T, r, vol, option_type="put")
    
    # Create DataFrames for Heatmaps
    call_df = pd.DataFrame(call_prices, index=stock_prices, columns=volatilities)
    put_df = pd.DataFrame(put_prices, index=stock_prices, columns=volatilities)
    
    # Plot Call Option Heatmap
    st.subheader("Call Option Price Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(call_df, cmap="YlGnBu", cbar=True, ax=ax)
    ax.set_xlabel("Volatility (σ)")
    ax.set_ylabel("Stock Price (S)")
    st.pyplot(fig)
    
    # Plot Put Option Heatmap
    st.subheader("Put Option Price Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(put_df, cmap="YlOrBr", cbar=True, ax=ax)
    ax.set_xlabel("Volatility (σ)")
    ax.set_ylabel("Stock Price (S)")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
