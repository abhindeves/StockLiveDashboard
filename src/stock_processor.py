import yfinance as yf
import numpy as np
import pandas as pd
from src.logger import get_logger
from src.utils import print_pred

class StockProcessor:
    def __init__(self, ticker, start_date, end_date):
        self.logger = get_logger()
        self.start_date = start_date
        self.end_date = end_date
        self.logger.info("Initializing StockProcessor for ticker: %s from %s to %s", ticker, start_date, end_date)
        self.stock_price = self.get_stock_data(ticker, start_date, end_date)

    def get_stock_data(self, ticker, start_date, end_date):
        self.logger.info("Downloading stock data for %s from %s to %s", ticker, start_date, end_date)
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        if not data.empty:
            self.logger.info("Stock data successfully retrieved for %s", ticker)
            self.logger.info("Combining today's stock data with previous 500 records to calculate indicators")
            stock_price_today_with_indicators = self.combine_old_stock_price(data)
        else:
            self.logger.warning("No data retrieved for %s", ticker)
            stock_price_today_with_indicators = None
            
        return stock_price_today_with_indicators

    def combine_old_stock_price(self, new_data):
        columns_to_load = new_data.columns
        file_path = "data/MASTER_MSFT_FINAL.csv"
        self.logger.info("Attempting to read old data from %s", file_path)
        
        try:
            old_data = pd.read_csv(file_path, usecols=columns_to_load, index_col=0).tail(500)
            self.logger.info("Old data successfully read from %s", file_path)
        except Exception as e:
            self.logger.error("Failed to read old data from %s: %s", file_path, e)
            exit(1)
        
        stock_price_old_new = self.calculate_indicators(pd.concat([old_data, new_data]))
        stock_price_today_with_indicators = stock_price_old_new.tail(1)
        stock_price_today_with_indicators.reset_index(inplace=True)
        return stock_price_today_with_indicators

    def calculate_indicators(self, stock_data):
        self.logger.info("Calculating indicators for stock data")
        stock_data['MA7'] = stock_data['Adj Close'].rolling(window=7).mean()
        stock_data['MA20'] = stock_data['Adj Close'].rolling(window=20).mean()
        stock_data['MACD'] = stock_data['Adj Close'].ewm(span=26).mean() - stock_data['Adj Close'].ewm(span=12, adjust=False).mean()
        stock_data['20SD'] = stock_data['Adj Close'].rolling(20).std()
        stock_data['upper_band'] = stock_data['MA20'] + (stock_data['20SD'] * 2)
        stock_data['lower_band'] = stock_data['MA20'] - (stock_data['20SD'] * 2)
        stock_data['EMA'] = stock_data['Adj Close'].ewm(com=0.5).mean()
        stock_data['logmomentum'] = np.log(stock_data['Adj Close'] - 1)
        stock_data.bfill(inplace=True)
        self.logger.info("Indicators calculated successfully")
        return stock_data

    def ouput_prediction(self, pred) -> None:
        self.logger.info("Outputting prediction for the period from %s to %s", self.start_date, self.end_date)
        print_pred(pred, self.start_date, self.end_date)