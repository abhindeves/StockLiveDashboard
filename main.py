# from src.stock_plotter import StockPlotter

# from src.prediction_model import PredictionModel


from src.logger import get_logger
from src.stock_processor import StockProcessor
from src.news_processor import NewsProcessor
from src.utils import data_combiner,prepare_final_data,append_new_data_to_existing,print_prediction
from src.prediction_model import PredictionModel

import yfinance as yf
import pandas as pd
import datetime


def main():
    logger = get_logger()
    logger.info("Logger has been initialized.")
    ticker = "MSFT"
    
    # start_date = datetime.datetime.now() - datetime.timedelta(days=1)
    # end_date = datetime.datetime.now()

    #TODO: DELETE THIS AFTER TESTING
    start_date = datetime.date(2024,8,30)

    end_date = datetime.date(2024,8,31)
    
    
    # Step 1: Get the stock data
    logger.info("Trying to get Stock Price")
    stock = StockProcessor(ticker, start_date, end_date)
    stock_price = stock.stock_price
    logger.info(f"Stock Price Details Aquired for {start_date}")
    if not stock.stock_price.empty: 
        # Step 2: Get News Data
        logger.info("Trying to get Stock News")
        processed_news = NewsProcessor(start_date,end_date,query='MSFT OR microsoft OR Stock OR Market')
        stock_news = processed_news.stock_news
        if processed_news != None:
            logger.info("Got Back News With Sentiment")
            logger.info("Sending to data_combiner to combine the news and price")
            combined_news_and_price = data_combiner(stock_price,stock_news)
            logger.info("Sending to prepare_final_df function to get only the required features")
            final_data_for_prediction = prepare_final_data(combined_news_and_price)
            print(final_data_for_prediction.dtypes)
            pred = PredictionModel(final_data_for_prediction)
            logger.info("Back to main with the prediction")
            append_new_data_to_existing(combined_news_and_price)
            print(pred.prediction)
            
        else:
            print("News data not available")
            logger.info(f"Stock News Data Not Available for {start_date}")
            exit(-1)
        # news_data = news_scraper.scrape_news()
    else:
        print("Stock Market is Closed Today")
        logger.info("Stock Market is Closed for Today")
        exit(-1)
 

    # # Step 3: Analyze sentiment of news
    # sentiment_analyzer = SentimentAnalyzer()
    # news_data['sentiment'] = news_data['summary'].apply(sentiment_analyzer.analyze_sentiment)

    # # Step 4: Prepare data for modeling
    # data_preparer = DataPreparation(stock_data, news_data)
    # merged_data = data_preparer.merge_data()

    # # Step 5: Load model and predict
    # model = PredictionModel(model_path="models/stock_movement_model.pkl")
    # features = merged_data.drop(columns=['Date', 'published', 'summary'])  # Example feature selection
    # prediction = model.predict_stock_movement(features)

    # # Step 6: Plot stock data
    # plotter = StockPlotter()
    # plotter.plot_stock_data(stock_data)

    # # Step 7: Update master data
    # model.update_master_data(merged_data, master_data_path="data/master_stock_data.csv")

    # print(f"Prediction for {ticker}: {prediction}")








if __name__ == "__main__":
    main()