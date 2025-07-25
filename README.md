# 📊 Modern Portfolio Theory with Fat-Tailed Asset Modeling

This project implements a robust and extensible framework for portfolio optimization under **Modern Portfolio Theory (MPT)**, extended to account for **fat-tailed return distributions** (e.g., using power-law distributions).

It supports backtesting, risk modeling, simulation-based optimization, and empirical analysis using both classical and heavy-tailed assumptions.

---

## 🔧 Features

- **Fat-Tailed Asset Modeling:**
  - Estimates **power-law exponents (α)** and **Taleb's Kappa** metrics.
  - Computes robust Value at Risk (VaR) and kurtosis for stress testing.
  - Compares fit of distributions (log-normal vs. power-law).

- **Modern Portfolio Theory (MPT):**
  - Simulates thousands of portfolios to identify optimal weights based on risk/return tradeoffs.
  - Supports both normal and power-law (fat-tailed) asset return modeling.
  - Includes plotting of efficient frontiers.

- **Robust Error Handling:**
  - Handles edge cases such as constant prices, non-numeric data, empty datasets, and missing values gracefully.

- **Data Pipeline:**
  - Imports stock data from Yahoo Finance.
  - Ticker validation from NASDAQ, S&P 500, Dow Jones, and Russell 3000.

- **Testing:**
  - Extensive **unit and integration tests** using `pytest`.
  - Covers all major edge cases, performance, and correctness of algorithms.

---

## 📁 Project Structure
```
├── modules/
│ ├── FatTailStock.py
│ ├── ModernPortfolioTheory.py 
│ └── utils.py
│
├── tests/
│ ├── unit/
│    ├── test_fattailstock.py 
│    └── test_modernportforliotheory.py
│ └── integration/
│    └── test_workflows.py
│
├── data/
│ ├── nasdaq.csv # List of NASDAQ tickers
│ ├── sp500.csv # S&P 500 tickers
│ ├── dow.csv # Dow Jones tickers
│ ├── russell.csv # Russell 3000 tickers
│ └── tickers.csv # FSE tickers (scraped)
│
├── pages/
│ ├── 01_Individual_Stock_Analysis.py
│ └── 02_Portfolio_Analysis.py
│
├── Home.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone [repo-link]
```
2. Install the ``requirements.txt`` using the command: 
``pip install -r requirements.txt``.

## How to run the app?

1. Navigate to the project folder in the terminal.
2. Run the folloing command:
Run he command:
```bash
streamlit run app.py
```