import yfinance as yf
import numpy as np
from src.logger import get_logger

class StockProcessor:
    def __init__(self,ticker,start_date,end_date):
        
        self.logger = get_logger()
        self.stock_price= self.get_stock_data(ticker,start_date,end_date)
        self.logger.info("StockProcessor Class initialisez")
        
                
    def get_stock_data(self,ticker,start_date,end_date):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d",progress=False)
            self.logger.info(f"Trying to get stock data for {start_date}")
        except Exception as e:
            self.logger.error("Stock market is closed today")
            data
        return data
            
    def calculate_indicators(self,stock_data):
        stock_data['MA7'] = stock_data['Adj Close'].rolling(window=7).mean()
        stock_data['MA20'] = stock_data['Adj Close'].rolling(window=20).mean()
        stock_data['MACD'] = stock_data['Adj Close'].ewm(span=26).mean() - stock_data['Adj Close'].ewm(span=12, adjust=False).mean()
        stock_data['20SD'] = stock_data['Adj Close'].rolling(20).std()
        stock_data['upper_band'] = stock_data['MA20'] + (stock_data['20SD'] * 2)
        stock_data['lower_band'] = stock_data['MA20'] - (stock_data['20SD'] * 2)
        stock_data['EMA'] = stock_data['Adj Close'].ewm(com=0.5).mean()
        stock_data['logmomentum'] = np.log(stock_data['Adj Close'] - 1)
        stock_data.fillna(method='bfill', inplace=True)
        return stock_data
        