import streamlit as st
import numpy as np
import yfinance as yf
from datetime import date
from modules.ModernPortfolioTheory import ModernPortfolioTheory as mpt
from pages.utils import load_ticker_data

st.title("🧠 Markowitz Portfolio Analysis")

st.markdown("""
We run Markowitz-based portfolio optimization for distribution-aware investment, In particular, we solve:""")
c1, c2 = st.columns(2)
with c1:
    st.write("For Normally-distributed Assets:")
    st.latex(r"""
    \begin{aligned}
        &\text{minimize}\ && w^T\Sigma w \\
        &\text{subject to}\ && \bar{r}w\geq r_{\min}, \\
        &&& \mathbf{1}^T w = 1,\quad w \succeq 0
    \end{aligned}
    """)
with c2:
    st.write("For Powerlaw-distributed Assets:")
    st.latex(r"""
    \begin{aligned}
        & \text{minimize} && \Sigma_{i^=1}^n w_i^{\mu}A_i^{\mu}\\
        &\text{subject to}\ &&\bar{r}w\geq r_{min}, \\
        &&& \mathbf{1}^Tw = 1,\ w\succeq  0.
    \end{aligned}
    """)

company_ticker, _, _, tickers = load_ticker_data()
tickers = company_ticker["ticker"].tolist()

st.sidebar.header("Simulation Parameters")
# Fetch Stock Data
train_start_date = st.sidebar.date_input(label="Training start", value=date(2020, 1, 1))
train_end_date = st.sidebar.date_input(label="Training end/Test start", value=date(2024, 1, 1))
test_end_date = st.sidebar.date_input(label="Test end", value=date(2025, 1, 1))
if train_end_date <= train_start_date:
    st.error("Training end date must be after training start date.")
n_paths = st.sidebar.slider("Number of Paths", min_value=500, max_value=10000, value=1000, step=500)
n_portfolios = st.sidebar.slider("Number of Portfolios", min_value=1000, max_value=25000, value=10000, step=1000)
big_loss = st.sidebar.slider("Portfolio Loss Tolerance (%)", min_value=10, max_value=80, value=20, step=1)

# Load data based on the input
try:
    stock_data = yf.download(tickers, start=train_start_date.strftime('%Y-%m-%d'),
                             end=test_end_date.strftime('%Y-%m-%d'))["Close"]
except Exception as e:
    st.warning(f"⚠️ Failed to download data: {e}")

stock_data = stock_data.dropna()
data_train = stock_data.loc[train_start_date: train_end_date]
data_test = stock_data.loc[train_end_date: test_end_date]

run = st.button("Run Markowitz Portfolio Analysis")

c1, c2 = st.columns(2)
if run:
    with c1:
        with st.spinner("🔄 Running the analysis, please wait..."):
            my_portfolio = mpt(data=data_train, asset_type="normal")
            my_portfolio.compute_mpt_params()
            fig1, ax1 = my_portfolio.plot_frontier()
            st.write("Training data results for normal assets:")
            st.pyplot(fig1)
            st.write(f"Return of normal (test data): {my_portfolio.compute_test_return(data_test): .3f}")
            # st.write(f"Optimal weights (obtained from the train data): {np.round(my_portfolio.w_opt, 3)}")
            company_ticker["opt_weights"] = np.round(my_portfolio.w_opt, 3)
            st.write(company_ticker[["company", "ticker", "opt_weights"]])
            st.success("✅ Analysis Complete")
    with c2:
        with st.spinner("🔄 Running the analysis, please wait..."):
            my_portfolio = mpt(data=data_train, asset_type="powerlaw")
            my_portfolio.compute_mpt_params()
            fig2, ax2 = my_portfolio.plot_frontier()
            st.write("Training data results for powerlaw assets:")
            st.pyplot(fig2)
            st.write(f"Return of powerlaw (test data): {my_portfolio.compute_test_return(data_test): .3f}")
            # st.write(f"Optimal weights (obtained from the train data): {np.round(my_portfolio.w_opt, 3)}")
            company_ticker["opt_weights"] = np.round(my_portfolio.w_opt, 3)
            st.write(company_ticker[["company", "ticker", "opt_weights"]])
            st.success("✅ Analysis Complete")
