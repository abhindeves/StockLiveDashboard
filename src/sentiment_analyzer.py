from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from textblob import TextBlob


class SentimentAnalyzer:
    def __init__(self,news_data) -> None: 
        self.news_with_sentiment = self.get_sentiment(news_data)


    def get_sentiment(self, news_data):
        SIA = SentimentIntensityAnalyzer()
        sentiment_columns = ['compound', 'neg', 'pos', 'neu']
        news_data = news_data.copy()  # Create a copy of the DataFrame
        news_data.loc[:, sentiment_columns] = news_data['title'].apply(lambda x: pd.Series(SIA.polarity_scores(' '.join(x))))
        # to get subjectivity and polarity
        news_data.loc[:, 'Subjectivity'] = news_data['title'].apply(get_subjectivity).values
        news_data.loc[:, 'Polarity'] = news_data['title'].apply(get_polarity).values
        return news_data
        
        
def get_subjectivity(text):
    s = ' '.join(text)
    return TextBlob(s).sentiment.subjectivity

# get polarity:

def get_polarity(text):
    s =  ' '.join(text)
    return TextBlob(s).sentiment.polarity

