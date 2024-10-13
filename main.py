from src.logger import get_logger
from src.stock_processor import StockProcessor
from src.news_processor import NewsProcessor
from src.utils import data_combiner,prepare_final_data,append_new_data_to_existing
from src.prediction_model import PredictionModel
import datetime

logger = get_logger()

def get_stock_data(ticker, start_date, end_date):
    logger.info("Trying to get Stock Price")
    stock = StockProcessor(ticker, start_date, end_date)
    if stock.stock_price.empty:
        logger.info("Stock Market is Closed for Today")
        raise ValueError("Stock Market is Closed Today")
    logger.info(f"Stock Price Details Acquired for {start_date}")
    return stock

def get_news_data(start_date, end_date, query):
    logger.info("Trying to get Stock News")
    processed_news = NewsProcessor(start_date, end_date, query)
    if processed_news.stock_news is None:
        logger.info(f"Stock News Data Not Available for {start_date}")
        raise ValueError("News data not available")
    logger.info("Got Back News With Sentiment")
    return processed_news

def main():
    logger = get_logger()
    logger.info("Logger has been initialized.")
    
    ticker = "MSFT"
    query = 'MSFT OR microsoft OR Stock OR Market'
    
    start_date = datetime.datetime.now() - datetime.timedelta(days=1)
    end_date = datetime.datetime.now()
    
    # start_date = datetime.date(2024,10,10)
    # end_date = datetime.date(2024,10,11)

    # Check if start_date is Friday, Saturday, or Sunday
    if start_date.weekday() >= 4:  # Friday is 4, Saturday is 5, Sunday is 6
        logger.info("Cannot predict for weekends or Friday")
        print("Cannot predict for weekends or Friday")
        return

    try:
        stock = get_stock_data(ticker, start_date, end_date)
        processed_news = get_news_data(start_date, end_date, query)
        
        combined_data = data_combiner(stock.stock_price, processed_news.stock_news)
        final_data = prepare_final_data(combined_data)
        
        prediction_model = PredictionModel(final_data)
        
        append_new_data_to_existing(combined_data)
        processed_news.ai_summary()
        stock.ouput_prediction(prediction_model.prediction)
    except ValueError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()