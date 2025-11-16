import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import date, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="MBA Portfolio Analyzer",
    page_icon="💼",
    layout="wide"
)

# --- Helper Functions ---

@st.cache_data
def get_stock_data(tickers, start_date, end_date):
    """Fetches close prices for a list of tickers."""
    try:
        # yf.download returns a DataFrame. We select 'Close'.
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        
        # If data is a Series (which can happen, e.g. single ticker), convert it to DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(tickers[0] if len(tickers) == 1 else 'Close')
        
        # If we have a single ticker and the column name is 'Close', rename it
        if len(tickers) == 1 and data.columns[0] == 'Close':
            data.columns = [tickers[0]]
            
        return data.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_returns(prices):
    """Calculates daily percentage returns."""
    return prices.pct_change().dropna()

# --- Title ---
st.title("📈 MBA Stock & Portfolio Analyzer")
st.markdown("""
    Welcome! This tool is designed for MBA students to explore the basics of portfolio analysis.
    Use the sidebar to select your stocks, set a date range, and assign portfolio weights.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("🛠️ Controls")

# Date Range
st.sidebar.subheader("1. Select Date Range")
end_date = date.today()
start_date_default = end_date - timedelta(days=365)

start_date = st.sidebar.date_input("Start Date", start_date_default)
end_date = st.sidebar.date_input("End Date", end_date)

if start_date > end_date:
    st.sidebar.error("Error: Start date must be before end date.")
    st.stop()

# Tickers
st.sidebar.subheader("2. Select Tickers")
default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
tickers = st.sidebar.multiselect(
    "Select one or more tickers",
    options=default_tickers + [t for t in ['NVDA', 'TSLA', 'AMZN', 'JPM', 'V', 'WMT'] if t not in default_tickers],
    default=default_tickers
)

if not tickers:
    st.warning("Please select at least one ticker to start.")
    st.stop()

# --- Data Fetching & Processing ---
prices = get_stock_data(tickers, start_date, end_date)

if prices.empty:
    st.error("No data fetched. Please check tickers and date range.")
    st.stop()

returns = calculate_returns(prices)

# Annualized mean returns and covariance matrix
# 252 trading days in a year
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

annual_mean_returns = mean_daily_returns * 252
annual_cov_matrix = cov_matrix * 252

# --- Sidebar for Weights ---
st.sidebar.subheader("3. Set Portfolio Weights")
st.sidebar.markdown("Enter weights for each asset. They will be auto-normalized.")

weights = {}
weight_inputs = {}
total_weight = 0

for ticker in tickers:
    weight_inputs[ticker] = st.sidebar.number_input(
        f"Weight {ticker}", 
        min_value=0.0, 
        value=1.0,  # Default to 1, will be normalized
        step=0.1
    )
    total_weight += weight_inputs[ticker]

# Normalize weights
if total_weight > 0:
    weights = {ticker: w / total_weight for ticker, w in weight_inputs.items()}
else:
    weights = {ticker: 0.0 for ticker in tickers} # Avoid division by zero

weights_array = np.array([weights[ticker] for ticker in tickers])

st.sidebar.subheader("Normalized Weights")
weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Normalized Weight'])
st.sidebar.dataframe(weights_df.style.format("{:.2%}"))


# --- Main Page with Tabs ---
tab1, tab2, tab3 = st.tabs([
    "📈 Price Time Series", 
    "📊 Asset Level Analysis", 
    "💼 Portfolio Level Analysis"
])

# --- Tab 1: Price Time Series ---
with tab1:
    st.header("Price Time Series")
    st.markdown("Observe the adjusted close price for your selected assets.")
    
    fig_price = go.Figure()
    for ticker in prices.columns:
        fig_price.add_trace(go.Scatter(
            x=prices.index, 
            y=prices[ticker], 
            name=ticker
        ))
    
    fig_price.update_layout(
        title="Close Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend_title_text="Tickers"
    )
    st.plotly_chart(fig_price, use_container_width=True)

# --- Tab 2: Asset Level Analysis ---
with tab2:
    st.header("Asset Level Analysis")
    st.markdown("Review the risk and return characteristics for each individual asset.")
    
    # --- Asset Statistics ---
    st.subheader("Asset Statistics (Annualized)")
    
    asset_stats = pd.DataFrame(index=tickers)
    asset_stats['Annual Mean Return'] = annual_mean_returns
    asset_stats['Annual Volatility (Std. Dev)'] = np.sqrt(np.diag(annual_cov_matrix))
    asset_stats['Variance'] = np.diag(annual_cov_matrix)
    
    st.dataframe(asset_stats.style.format({
        "Annual Mean Return": "{:.2%}",
        "Annual Volatility (Std. Dev)": "{:.2%}",
        "Variance": "{:.4f}"
    }))

    # --- PDF of Returns ---
    st.subheader("Probability Distribution of Daily Returns")
    st.markdown("This shows the distribution (PDF) of daily returns for each asset. Taller, narrower curves mean less volatility.")
    
    hist_data = [returns[ticker] for ticker in tickers]
    group_labels = tickers
    
    fig_pdf = ff.create_distplot(
        hist_data, 
        group_labels, 
        bin_size=0.005, 
        show_hist=False, 
        show_rug=False
    )
    
    fig_pdf.update_layout(
        title="Distribution of Daily Returns",
        xaxis_title="Daily Return",
        yaxis_title="Density",
        hovermode="x unified" # Add vertical line hover
    )
    st.plotly_chart(fig_pdf, use_container_width=True)

# --- Tab 3: Portfolio Level Analysis ---
with tab3:
    st.header("Portfolio Level Analysis")
    st.markdown("Review the risk and return characteristics for your combined portfolio based on the weights you set.")

    # --- Portfolio Calculations ---
    portfolio_daily_returns = (returns * weights_array).sum(axis=1)
    
    # Portfolio Mean Return (Annualized)
    portfolio_mean_return = np.dot(weights_array, annual_mean_returns)
    
    # Portfolio Variance & Volatility (Annualized)
    portfolio_variance = np.dot(weights_array.T, np.dot(annual_cov_matrix, weights_array))
    portfolio_volatility = np.sqrt(portfolio_variance)

    # --- Display Portfolio Statistics ---
    st.subheader("Portfolio Statistics (Annualized)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio Mean Return", f"{portfolio_mean_return:.2%}")
    col2.metric("Portfolio Volatility (Std. Dev)", f"{portfolio_volatility:.2%}")
    col3.metric("Portfolio Variance", f"{portfolio_variance:.4f}")

    # --- Cumulative Portfolio Returns ---
    st.subheader("Portfolio Cumulative Returns")
    st.markdown("This chart shows the growth of $1 invested in your portfolio compared to each individual asset.")
    
    # Calculate cumulative returns for portfolio
    portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod()
    
    # Calculate cumulative returns for individual assets
    asset_cumulative_returns = (1 + returns).cumprod()
    
    fig_port_cum = go.Figure()
    
    # Add Portfolio Trace
    fig_port_cum.add_trace(go.Scatter(
        x=portfolio_cumulative_returns.index,
        y=portfolio_cumulative_returns,
        name="Portfolio (Your Weights)",
        line=dict(color='blue', width=4) # Make portfolio line stand out
    ))
    
    # Add Individual Asset Traces
    for ticker in tickers:
        fig_port_cum.add_trace(go.Scatter(
            x=asset_cumulative_returns.index,
            y=asset_cumulative_returns[ticker],
            name=ticker,
            line=dict(dash='dot', width=1.5), # Make asset lines thinner and dotted
            opacity=0.8
        ))
    
    fig_port_cum.update_layout(
        title="Portfolio vs. Individual Asset Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (Growth of $1)",
        hovermode="x unified", # This enables the vertical line on mouseover
        legend_title_text="Assets"
    )
    st.plotly_chart(fig_port_cum, use_container_width=True)

    # --- PDF of Portfolio Returns ---
    st.subheader("Probability Distribution of Portfolio vs. Asset Daily Returns")
    st.markdown("Compare the portfolio's return distribution (blue) against its individual components. A narrower portfolio curve indicates diversification benefits (lower volatility).")
    
    # Create list of all data series for the plot
    hist_data_all = [portfolio_daily_returns] + [returns[ticker] for ticker in tickers]
    group_labels_all = ['Portfolio (Your Weights)'] + tickers
    
    fig_port_pdf = ff.create_distplot(
        hist_data_all, 
        group_labels_all, 
        bin_size=0.005, 
        show_hist=False, 
        show_rug=False
    )
    
    # Optional: Make the portfolio line stand out more
    fig_port_pdf.data[0].line.color = 'blue'
    fig_port_pdf.data[0].line.width = 4
    for i in range(1, len(fig_port_pdf.data)):
        fig_port_pdf.data[i].line.dash = 'dot'
        fig_port_pdf.data[i].line.width = 1.5
        fig_port_pdf.data[i].opacity = 0.8

    fig_port_pdf.update_layout(
        title="Distribution of Portfolio vs. Asset Daily Returns",
        xaxis_title="Daily Return",
        yaxis_title="Density",
        hovermode="x unified" # Add vertical line hover
    )
    st.plotly_chart(fig_port_pdf, use_container_width=True)