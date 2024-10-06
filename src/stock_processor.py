import yfinance as yf
import numpy as np
import pandas as pd
from src.logger import get_logger

class StockProcessor:
    def __init__(self,ticker,start_date,end_date):
        
        self.logger = get_logger()
        self.logger.info("StockProcessor Class initialisez")
        self.stock_price= self.get_stock_data(ticker,start_date,end_date)
        
        
                
    def get_stock_data(self,ticker,start_date,end_date):

        data = yf.download(ticker, start=start_date, end=end_date, interval="1d",progress=False)
        if not data.empty:
            self.logger.info(f"Trying to get stock data for {start_date}")
            self.logger.info("Combine todays stock data with previous 500 data to calculate matrix")
            stock_price_today_with_indicators = self.combine_old_stock_price(data)
        else:
            stock_price_today_with_indicators = None
            
        return stock_price_today_with_indicators
         
         
    def combine_old_stock_price(self,new_data):
        # clean the new stock price
        columns_to_load = new_data.columns
        file_path = "data\MASTER_MSFT_FINAL.csv"
        self.logger.info(f"Trying to read OLD data from {file_path}")
        
        try:
            old_data = pd.read_csv(file_path, usecols=columns_to_load, index_col=0).tail(500)
        except Exception as e:
            self.logger.error("Failed to read old files")
            exit(1)
        
        self.logger.info("OLD Data read from the path")
        stock_price_old_new = self.calculate_indicators(pd.concat([old_data,new_data]))
        stock_price_today_with_indicators = stock_price_old_new.tail(1)
        stock_price_today_with_indicators.reset_index(inplace=True)
        return stock_price_today_with_indicators

    
       
    def calculate_indicators(self,stock_data):
        stock_data['MA7'] = stock_data['Adj Close'].rolling(window=7).mean()
        stock_data['MA20'] = stock_data['Adj Close'].rolling(window=20).mean()
        stock_data['MACD'] = stock_data['Adj Close'].ewm(span=26).mean() - stock_data['Adj Close'].ewm(span=12, adjust=False).mean()
        stock_data['20SD'] = stock_data['Adj Close'].rolling(20).std()
        stock_data['upper_band'] = stock_data['MA20'] + (stock_data['20SD'] * 2)
        stock_data['lower_band'] = stock_data['MA20'] - (stock_data['20SD'] * 2)
        stock_data['EMA'] = stock_data['Adj Close'].ewm(com=0.5).mean()
        stock_data['logmomentum'] = np.log(stock_data['Adj Close'] - 1)
        stock_data.bfill(inplace=True)
        return stock_data
        