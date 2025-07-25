"""
utils.py

Shared utility functions for the multi-page Streamlit app:
- Loading and managing company ticker data
- Managing sidebar inputs for date ranges
- Adding/removing companies from the ticker list

Assumes the presence of:
- A CSV file at "data/company_ticker.csv" with columns: ["company", "ticker", "purchase_date"]
"""

import streamlit as st
import pandas as pd
from typing import Tuple, Dict, List

import plotly.graph_objects as go


def load_ticker_data() -> Tuple[pd.DataFrame, Dict[str, str], List[str], List[str]]:
    """
    Loads company ticker data from the CSV file and returns structured data.

    Returns:
        Tuple containing:
            - DataFrame: the full ticker table
            - Dict[str, str]: mapping from company names to tickers
            - List[str]: list of company names
            - Dict[str, str]: mapping from company names to purchase dates
    """
    csv_file = "./data/company_ticker.csv"
    company_ticker = pd.read_csv(csv_file)

    if "purchase_date" not in company_ticker.columns:
        company_ticker["purchase_date"] = ""

    tickers: List[str] = company_ticker["ticker"].tolist()
    company_names: List[str] = company_ticker["company"].tolist()

    ticker_dict: Dict[str, str] = dict(zip(company_names, tickers))

    return company_ticker, ticker_dict, company_names, tickers


def modify_company_list_ui(company_ticker: pd.DataFrame) -> None:
    """
    Creates sidebar UI for adding or deleting companies from the ticker list.
    Automatically saves updates back to the CSV.

    Args:
        company_ticker (pd.DataFrame): DataFrame containing company/ticker data
    """
    st.sidebar.subheader("Manage Company List")
    csv_file = "../data/company_ticker.csv"

    new_company: str = st.sidebar.text_input("Company")
    new_ticker: str = st.sidebar.text_input("Ticker")

    col1, col2 = st.sidebar.columns(2)

    if col1.button("Add"):
        if new_company and new_ticker and new_company not in company_ticker["company"].values:
            new_entry = pd.DataFrame({
                "company": [new_company],
                "ticker": [new_ticker],
                "purchase_date": [""]
            })
            company_ticker = pd.concat([new_entry, company_ticker], ignore_index=True)
            company_ticker.to_csv(csv_file, index=False)
            st.sidebar.success(f"✅ Added {new_company}!")
            st.rerun()

    if col2.button("Delete"):
        idx = company_ticker["company"] == new_company
        if idx.any():
            company_ticker = company_ticker[~idx]
            company_ticker.to_csv(csv_file, index=False)
            st.sidebar.success(f"🗑️ Deleted {new_company}!")
            st.rerun()


def get_date_range(unit: str) -> int:
    """
    Provides a sidebar slider to select a time range in days based on a unit.

    Args:
        unit (str): One of ["day", "week", "month"]

    Returns:
        int: Number of days selected
    """
    if unit == "day":
        return st.sidebar.slider("Past Days", 7, 365, 28)
    elif unit == "week":
        weeks = st.sidebar.slider("Past Weeks", 1, 52, 12)
        return weeks * 7
    elif unit == "month":
        months = st.sidebar.slider("Past Months", 1, 24, 6)
        return months * 30
    else:
        st.sidebar.error("Invalid time unit selected.")
        return 30

def plot_ts(data_list: list):
    """
    Plots the time series data in the streamlit format.

    Args:
        A list of all the time series dataframes to be plotted.
    """

    fig = go.Figure()
    for data in data_list:
        column_name = data.columns[0]
        fig.add_trace(go.Scatter(
            x=data.index, y=data[column_name],
            mode='lines', name=data.columns[0],
            hovertemplate="Date: %{x}<br>Return: %{y:.2%}"
        ))
    fig.update_layout(
        title="Training, Test, & Model Prediction",
        xaxis_title="Date",
        yaxis_title="Value",
    )
    st.plotly_chart(fig)
