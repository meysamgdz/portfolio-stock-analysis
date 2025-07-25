import plotly.graph_objects as go
from datetime import datetime, timedelta
from modules.FatTailStock import FatTailStock
import yfinance as yf
from pages.utils import *
from modules.utils import *

# Streamlit App Title
st.title("📈 Stock & Portfolio Analysis")

# Sidebar - Company Selection
csv_file = "data/company_ticker.csv"
company_ticker = pd.read_csv(csv_file, sep=",")
if "purchase_date" not in company_ticker.columns:
    company_ticker["purchase_date"] = ""

tickers = company_ticker["ticker"].tolist()
company_names = company_ticker["company"].tolist()
ticker_dict = dict(zip(company_names, tickers))
purchase_dates_dict = dict(zip(company_names, company_ticker["purchase_date"]))

# --- STOCK DATA ANALYSIS ---
st.sidebar.subheader("Stock Analysis Settings")
selected_company = st.sidebar.selectbox("Choose a company", company_names)
time_unit = st.sidebar.radio("Select Time Unit:", ["day", "week", "month"], horizontal=True)
if time_unit == "day":
    num_days = st.sidebar.slider("Select number of past days", min_value=7, max_value=365, value=132, step=1)
elif time_unit == "week":
    num_days = st.sidebar.slider("Select number of past weeks", min_value=1, max_value=100, value=54, step=1)*5
if time_unit == "month":
    num_days = st.sidebar.slider("Select number of past months", min_value=7, max_value=240, value=24, step=1)*22

# --- ADD/DELETE COMPANY SECTION ---
st.sidebar.subheader("Modify Portfolio List")
new_company = st.sidebar.text_input("Company")
new_ticker = st.sidebar.text_input("Ticker")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("➕ Add"):
        if new_company and new_ticker:
            if new_company not in company_names and new_ticker not in tickers:
                new_entry = pd.DataFrame({"company": [new_company], "ticker": [new_ticker], "purchase_date": [""]})
                company_ticker = pd.concat([new_entry, company_ticker], ignore_index=True)
                company_ticker.to_csv(csv_file, index=False, sep=",")
                st.sidebar.success(f"Added {new_company} ({new_ticker}) successfully!")
                st.rerun()
            else:
                st.sidebar.warning("⚠️ Company or Ticker already exists!")
with col2:
    if st.button("🗑️ Del"):
        if new_company in company_names:
            company_ticker = company_ticker[company_ticker["company"] != new_company]
            company_ticker.to_csv(csv_file, index=False, sep=",")
            st.sidebar.success(f"Deleted {new_company} successfully!")
            st.rerun()
        elif new_ticker in tickers:
            company_ticker = company_ticker[company_ticker["ticker"] != new_ticker]
            company_ticker.to_csv(csv_file, index=False, sep=",")
            st.sidebar.success(f"Deleted ticker {new_ticker} successfully!")
            st.rerun()
        else:
            st.sidebar.warning("⚠️ Company not found!")
            st.rerun()

# Fetch Stock Data
end_date = datetime.now()
start_date = end_date - timedelta(days=num_days)
ticker = ticker_dict[selected_company]

# Determine reference purchase date
purchase_date_str = str(purchase_dates_dict.get(selected_company, "")).strip()
try:
    purchase_date = datetime.strptime(purchase_date_str, "%Y-%m-%d") if purchase_date_str else end_date - timedelta(
        days=1)
except ValueError:
    purchase_date = end_date - timedelta(days=1)  # Default to yesterday if the date is invalid

# Ensure purchase date is within graph range
if purchase_date < start_date:
    start_date = purchase_date  # Expand the graph range if needed

stock_data = pd.DataFrame()
try:
    downloaded_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'),
                                  end=end_date.strftime('%Y-%m-%d'))
    stock_data[ticker] = downloaded_data["Close"]
except Exception as e:
    st.warning(f"⚠️ Failed to download data for {selected_company} ({ticker}): {e}")

# Run Fat-Tail Analysis
if not stock_data.empty:

    st.subheader("Ticker Index Checker")

    if ticker:
        status = check_ticker(ticker)
        # Display results in a horizontal row
        cols = st.columns(len(status))
        for col, (index_name, is_member) in zip(cols, status.items()):
            if is_member:
                col.markdown(f"**{index_name}**: ✅")
            else:
                col.markdown(f"**{index_name}**: ❌")

    ftd = FatTailStock(stock_data)
    ftd.get_fat_tail_metrics()

    # Get daily returns for waterfall chart
    stock_data["daily_change"] = stock_data[ticker].diff().fillna(0)

    # Get reference prices
    first_price = stock_data[ticker].iloc[0]  # First date in the graph

    # Plot Data
    st.subheader("📈 Individual Stock Visualization")
    chart_option = st.radio("Select Chart Type:", ["Price & Returns", "Waterfall Chart"], horizontal=True)

    if chart_option == "Price & Returns":
        # Twin-Axis Chart (Stock Price & Returns)
        fig = go.Figure()

        # Add Adjusted Close Price (Primary Axis)
        fig.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data[ticker],
            mode='lines', name=f"{selected_company} - Price",
            line=dict(color='blue'),
            hovertemplate="Date: %{x}<br>Price: %{y:.2f}€"
        ))

        # Add Linear Returns (Secondary Axis)
        lin_returns = ftd.get_lin_returns()
        fig.add_trace(go.Scatter(
            x=lin_returns.index, y=lin_returns[ticker],
            mode='lines', name=f"{selected_company} - Returns",
            line=dict(color='purple'),
            yaxis="y2",
            hovertemplate="Date: %{x}<br>Return: %{y:.2%}"
        ))

        # Layout with twin axis
        fig.update_layout(
            title="Stock Price & Returns",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Close Price", side="left"),
            yaxis2=dict(title="Returns", overlaying="y", side="right"),
            hovermode="x"
        )

        st.plotly_chart(fig)

    else:
        # Waterfall Chart (Stock Price Changes)
        fig_waterfall = go.Figure()

        fig_waterfall.add_trace(go.Waterfall(
            x=stock_data.index.strftime('%Y-%m-%d'),
            y=stock_data["daily_change"],
            base=first_price,  # Start at the first recorded price
            increasing=dict(marker=dict(color="green")),
            decreasing=dict(marker=dict(color="red")),
            totals=dict(marker=dict(color="blue")),
            hoverinfo="x+y"
        ))

        fig_waterfall.update_layout(
            title="Stock Price Changes (Waterfall Chart)",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Stock Price (€)"),
            hovermode="x"
        )

        st.plotly_chart(fig_waterfall)

else:
    st.error("❌ No data available. Please select a valid company.")

# Display metrics
st.subheader("📊 Fat-Tail Metrics")
c1, c2, c3 = st.columns(3)
with c1:
    st.write(r"Powerlaw ($\alpha$, $x_{min}$):", ftd.alpha)
with c2:
    st.write(r"Lognorm ($\mu$, $e^{\mu}$, $\sigma$):", ftd.lognorm_params)
with c3:
    # st.write("VaR (95% conf.):", ftd.value_at_risk(0.95))
    st.write("Taleb Kappa:\n", ftd.t_kapa.loc["t_kappa", ticker])
    st.write("Kurtosis (Excess):\n", ftd.kurtosis[ticker])
st.markdown(ftd.compare_powerlaw(), unsafe_allow_html=True)
fig, ax = ftd.plot_distribution_fits()
st.pyplot(fig)