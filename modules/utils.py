import os
import pandas as pd
import requests
from bs4 import BeautifulSoup

def check_ticker(ticker: str) -> dict:
    """
    Check whether a given stock ticker exists in major U.S. stock indices.

    Args:
        ticker (str): The stock ticker symbol (case-insensitive).

    Returns:
        dict: A dictionary indicating whether the ticker is present in each of the following:
            - 'NASDAQ'
            - 'S&P 500'
            - 'Dow Jones'
            - 'Russell 3000'
    """
    ticker = ticker.upper()
    nasdaq_tickers = list(pd.read_csv("./data/nasdaq.csv", sep=",")["ticker"])
    sp500_tickers = list(pd.read_csv("./data/sp500.csv", sep=",")["ticker"])
    dow_tickers = list(pd.read_csv("./data/dow.csv", sep=",")["ticker"])
    russell_tickers = list(pd.read_csv("./data/russell.csv", sep=",")["ticker"])
    status = {
        'NASDAQ': ticker in nasdaq_tickers,
        'S&P 500': ticker in sp500_tickers,
        'Dow Jones': ticker in dow_tickers,
        'Russell 3000': ticker in russell_tickers
    }
    return status

def get_fse_tickers() -> pd.DataFrame:
    """
    Retrieve a list of Frankfurt Stock Exchange tickers.

    Checks for a local CSV file named 'tickers'. If not found, scrapes the listings from the DividendMax website.

    Returns:
        pandas.DataFrame: A DataFrame with columns ['company', 'ticker'].
    """
    filename = "tickers"
    if check_csv_file_exists(filename):
        return pd.read_csv(filename)
    else:
        return scrape_frankfurt_stock_exchange_listings()

def check_csv_file_exists(filename: str) -> bool:
    """
    Check if a CSV file exists in the current working directory.

    Args:
        filename (str): The filename to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, filename)
    return os.path.isfile(file_path)

def split_list(input_list: list, sublist_size: int) -> list:
    """
    Split a list into sublists of a specified maximum size.

    Args:
        input_list (list): The list to split.
        sublist_size (int): The maximum size of each sublist.

    Returns:
        list: A list of sublists.
    """
    return [input_list[i:i + sublist_size] for i in range(0, len(input_list), sublist_size)]

def scrape_frankfurt_stock_exchange_listings() -> pd.DataFrame:
    """
    Scrape company and ticker listings from the Frankfurt Stock Exchange via DividendMax.

    Iterates through multiple pages of listings and saves the result as 'tickers.csv'.

    Returns:
        pandas.DataFrame: A DataFrame containing company names and their tickers.
    """
    base_url = 'https://www.dividendmax.com/stock-exchange-listings/germany/frankfurt-stock-exchange'
    data = []
    for page in range(1, 7):
        url = f'{base_url}?page={page}'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')

        for row in table.find_all('tr')[1:]:  # Skip header row
            columns = row.find_all('td')
            company = columns[0].text.strip()
            ticker = columns[1].text.strip()
            price = columns[2].text.strip()
            market_cap = columns[3].text.strip()
            indices = columns[4].text.strip()
            data.append([company, ticker])

    df = pd.DataFrame(data, columns=['company', 'ticker'])
    df.to_csv('tickers.csv', index=False)
    return df
