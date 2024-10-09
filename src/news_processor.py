from pygooglenews import GoogleNews
import pandas as pd
import datetime as dt

from src.logger import get_logger
from src.sentiment_analyzer import SentimentAnalyzer
from src.utils import summarize


class NewsProcessor:
    def __init__(self, start_date, end_date, query):
        self.logger = get_logger()
        self.start_date = start_date
        self.end_date = end_date
        self.logger.info("Initializing News Processor with start date: %s and end date: %s", start_date, end_date)
        self.stock_news = self.scrape_news(query, start_date, end_date)
    
# Function to scrape news articles using Google News
    def scrape_news(self, query, start_date, end_date):
        self.logger.info("Starting news scraping for query: %s", query)
        gn = GoogleNews()
        all_news = pd.DataFrame(columns=['title', 'published', 'source'])
        current_date = pd.to_datetime(start_date)
        while current_date <= pd.to_datetime(end_date):
            try:
                self.logger.debug("Fetching news for date: %s", current_date.date())
                result = gn.search(query, from_=str(current_date), to_=str(current_date + pd.Timedelta(days=1)))
                df = pd.DataFrame(result['entries'])[['title', 'published', 'source']]
                if df.empty:
                    self.logger.warning("No news found for date: %s", current_date.date())
                    df = pd.DataFrame({'title': [''], 'published': [current_date.date()], 'source': ['']})
                else:
                    df['published'] = pd.to_datetime(df['published']).dt.date
                all_news = pd.concat([all_news, df], ignore_index=True)
            except Exception as e:
                self.logger.error("Error occurred while fetching news data for date %s: %s", current_date.date(), e)
                return None
            current_date += pd.Timedelta(days=1)
        all_news.sort_values(by='published', inplace=True)
        all_news.rename(columns={'published': 'Date'}, inplace=True)
        self.logger.info("Completed scraping news. Total articles fetched: %d", len(all_news))
        
        all_news = self.process_news_data(all_news)
        self.logger.info("Processed news data with sentiment analysis.")
        return all_news
    
    def process_news_data(self, stock_news):
        stock_news = pd.DataFrame(stock_news)
        stock_news = stock_news.groupby('Date').agg({'title': list})
        stock_news.reset_index(inplace=True)
        target = self.start_date
        stock_news_cleaned = stock_news[stock_news['Date'] == target]
        self.logger.info("Cleaned news data for target date: %s", target)
        self.logger.info("Initializing Sentiment Analyzer")
        final_news = SentimentAnalyzer(stock_news_cleaned)
        self.logger.info("Sentiment analysis completed. Returning results.")
        return final_news.news_with_sentiment
        
    def ai_summary(self) -> str:
        my_list = self.stock_news['title'].values[0]
        s = ';'.join(my_list)
        summary = summarize(s)
        self.logger.info("Generating summary for stock news on date: %s", self.start_date)
        print("\nSTOCK NEWS SUMMARY ON {} \n".format(self.start_date))
        print(summary)

