from pygooglenews import GoogleNews
import pandas as pd
import datetime as dt

from src.logger import get_logger
from src.sentiment_analyzer import SentimentAnalyzer


class NewsProcessor:
    def __init__(self,start_date,end_date,query):
        self.logger = get_logger()
        self.start_date = start_date
        self.end_Date = end_date
        self.logger.info("Initializing News Processor")
        self.stock_news = self.scrape_news(query,start_date,end_date)
    
# Function to scrape news articles using Google News
    def scrape_news(self,query,start_date, end_date):
        self.logger.info("Inside scrape_news_module")
        self.logger.info("Initializing GoogleNews object")
        gn = GoogleNews()
        all_news = pd.DataFrame(columns=['title', 'published', 'source'])
        current_date = pd.to_datetime(start_date)
        while current_date <= pd.to_datetime(end_date):
            try:
                result = gn.search(query, from_=str(current_date), to_=str(current_date + pd.Timedelta(days=1)))
                df = pd.DataFrame(result['entries'])[['title', 'published', 'source']]
                if df.empty:
                    df = pd.DataFrame({'title': [''], 'published': [current_date.date()], 'source': ['']})
                else:
                    df['published'] = pd.to_datetime(df['published']).dt.date
                all_news = pd.concat([all_news, df], ignore_index=True)
            except Exception as e:
                print(f"Error occurred while fetching news data: {e}")
                return None
            current_date += pd.Timedelta(days=1)
        all_news.sort_values(by='published', inplace=True)
        all_news.rename(columns={'published':'Date'},inplace=True)
        self.logger.info("Passing the raw news data for cleaning to process_news_data function")
        
        all_news = self.process_news_data(all_news)
        self.logger.info("News Data with Sentimet is in scarper_news function")
        self.logger.info("Passing the New Data with Sentiment to News Processor Class")
        return all_news
    
    def process_news_data(self,stock_news):
        stock_news = pd.DataFrame(stock_news)
        stock_news = stock_news.groupby('Date').agg({'title':list})
        stock_news.reset_index(inplace=True)
        target = (self.start_date)
        stock_news_cleaned = stock_news[stock_news['Date'] == target]
        self.logger.info("News Data Cleaned, Data will be sent for sentiment analysis")
        self.logger.info("Initializing Sentiment Analizer")
        final_news = SentimentAnalyzer(stock_news_cleaned)
        self.logger.info("News Data with Sentiment is returned to process_new_data function")
        return final_news.news_with_sentiment
        
        
