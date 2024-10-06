import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pygooglenews import GoogleNews
from datetime import datetime
from datetime import timedelta
import joblib
import yfinance as yf
from colorama import init, Fore, Style
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

# Function to scrape news articles using Google News
def scrape_news(start_date, end_date):
    gn = GoogleNews()
    all_news = pd.DataFrame(columns=['title', 'published', 'source'])
    current_date = pd.to_datetime(start_date)
    while current_date <= pd.to_datetime(end_date):
        result = gn.search('MSFT OR microsoft OR Stock OR Market', from_=str(current_date), to_=str(current_date + pd.Timedelta(days=1)))
        df = pd.DataFrame(result['entries'])
        if df.empty:
            df = pd.DataFrame({'title': [''], 'published': [current_date], 'source': ['']})
        else:
            df = df[['title', 'published', 'source']]
            df['published'] = pd.to_datetime(df['published']).dt.date
        all_news = pd.concat([all_news, df], ignore_index=True)
        current_date += pd.Timedelta(days=1)
    all_news.sort_values(by='published', inplace=True)
    all_news.rename(columns={'published':'Date'},inplace=True)
    return all_news

def get_subjectivity(text):
    s = ' '.join(text)
    return TextBlob(s).sentiment.subjectivity

# get polarity:

def get_polarity(text):
    s =  ' '.join(text)
    return TextBlob(s).sentiment.polarity

# Function to calculate sentiment analysis
def calculate_sentiment(news_data):
    sia = SentimentIntensityAnalyzer()
    sentiment_columns = ['compound', 'neg', 'pos', 'neu']
    news_data[sentiment_columns] = news_data['title'].apply(lambda x: pd.Series(sia.polarity_scores(' '.join(x))))
    # to get subjectivity and polarity
    news_data['Subjectivity'] = news_data['title'].apply(get_subjectivity)
    news_data['Polarity'] = news_data['title'].apply(get_polarity)
    return news_data

# Function to get technical indicators for stock data
def get_technical_indicators(stock_data):
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

def join_data(stock_data, news_data):
    # Shift the 'price' column by one day to get the previous day's price
    #stock_data['price_pred'] = stock_data['Adj Close'].shift(1)

    # Compare today's price with the previous day's price
    #stock_data['movement'] = stock_data.apply(lambda row: 1 if row['Adj Close'] > row['price_pred'] else 0 , axis=1)
    # Drop the 'previous_price' column if you don't need it anymore
    #stock_data.drop(columns=['price_pred'], inplace=True)
    return stock_data,news_data


# Function to prepare data for modeling
def prepare_data(merged_data):
    # merged_data = pd.merge(left=stock_data, right=news_data, on='Date')
    #merged_data.set_index('Date',inplace=True)
    X = merged_data[['Open', 'High', 'Close', 'Subjectivity', 'Polarity', 'compound','neg', 'pos', 'neu', 'MA7', 
                     'MA20', 'MACD', '20SD', 'upper_band','lower_band', 'EMA', 'logmomentum']]

    #y = merged_data['movement']
    return X

# Function to summarise todays Stock news
def summarize_stock_news(s):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        messages=[
            {"role": "system", "content": "You are a highly skilled AI trained in Stock Market News Summarization. I would like you to read the list of News headlines and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the overall market trend without needing to read the entire text. You can start by greeting the Day trader in an appealing manner. Please avoid unnecessary details or tangential points."},
            {"role": "user", "content": s}
        ]
    )

    return completion.choices[0].message.content

# Plot for stock data analysis
def plot_stock_data():
    # Pull data from Yahoo Finance for MSFT from 2020 Jan to today
    stock_data = yf.download("MSFT", start="2020-01-01",progress=False)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'],
                    mode='lines'))
    title = []
    title.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Analyze Yourself!! MSFT Stock Data',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
    fig1.update_layout(xaxis_title='Date',
                       yaxis_title='Close stock value',
                      annotations=title)
    fig1.show()



# # Get today's date as a string
# today_date_str = datetime.now().strftime('%Y-%m-%d')

# # Convert today's date string to a datetime object
# today_date = datetime.strptime(today_date_str, '%Y-%m-%d')

# # Calculate one day ago
# one_day_ago = today_date - timedelta(days=1)

# # Convert one day ago to a string in the same format as today_date_str
# one_day_ago_str = one_day_ago.strftime('%Y-%m-%d')

print("\nMODEL AND MASTER DATA LOADED SUCCESSFULLY........\n")

# # Get today's date
today = datetime.today()

# Get tomorrow's date by adding one day to today's date
next_day = today + timedelta(days=1)

# Convert the dates to strings in the format "%Y-%m-%d"
start_date = today.strftime("%Y-%m-%d")
end_date = next_day.strftime("%Y-%m-%d")


# start_date = '2024-04-22'

# end_date = '2024-04-23'

data = yf.download("MSFT", start=start_date, end=end_date, interval="1d",progress=False)

# If data is available, print a message indicating the market is open
if not data.empty:
    # Scrape news data for one day ago
    news_data = scrape_news(start_date, start_date)
    news_data = news_data.reset_index()
    news_data['Date'] = pd.to_datetime(news_data['Date'])
    news_data = news_data[news_data['Date'] == start_date]


    # Load stock data for today
    stock_data = yf.download("MSFT", start=start_date, end=end_date, interval="1d")

    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = get_technical_indicators(stock_data)
    print("\nSTOCK PRICE DATA AND STOCK NEWS DATA LOADED SUCCESSFULLY..........\n")


    # clean the new stock price
    columns_to_load = stock_data.columns
    file_path = "/kaggle/working/MASTER_UPD.csv"
    all_data = pd.read_csv(file_path,usecols=columns_to_load,index_col=[0])
    data = all_data.tail(100)
    data = get_technical_indicators(pd.concat([data,stock_data]))
    stock_data = data.tail(1)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # clean the NEWS Data
    news_data = news_data.groupby('Date').agg({'title':list})
    news_data = calculate_sentiment(news_data)
    news_data = news_data.reset_index()
    news_data['Date'] = pd.to_datetime(news_data['Date'])

    # # Join both data
    merged_data = pd.merge(left=stock_data,right=news_data,on='Date')
    # merged_data.set_index('Date',inplace=True)
    merged_data_to_predict = prepare_data(merged_data)


    loaded_model = joblib.load('/kaggle/input/firstmodel/best_model_logreg.pkl')


    prediction_tomorrow = loaded_model.predict(merged_data_to_predict.fillna(value=0))
    pred = prediction_tomorrow[0]

    # Add todays data to the table in order to plot graphs
    # Read master data
    master_data = pd.read_csv("/kaggle/working/MASTER_UPD.csv", index_col=[0])
    master_data = master_data.reset_index()
    master_data['Date'] = pd.to_datetime(master_data['Date'])

    # Check if the date from merged dataset is already in master data
    merged_date = merged_data['Date']
    if not merged_date.isin(master_data['Date']).any():
        # If date is not present, concatenate the merged data
        master_data = pd.concat([master_data, merged_data.reset_index()], ignore_index=True)
        master_data.to_csv('MASTER_UPD.csv', index=False)
    else:
        pass


    # -------------------------PRINTING THE OUTPUT-----------------------#

    # Initialize colorama to enable colored text
    init(autoreset=True)

    # Assuming pred is either "Up" or "Down"
    if pred == 1:
        prediction_statement = f"{Fore.GREEN}The stock price is predicted to go UP on {end_date}.{Style.RESET_ALL}"
    elif pred == 0:
        prediction_statement = f"{Fore.RED}The stock price is predicted to go DOWN {end_date}.{Style.RESET_ALL}"

    print("=" * 50)
    print(f"{Fore.YELLOW}       📈📉 Stock Price Prediction 📈📉{Style.RESET_ALL}")
    print("=" * 50)
    print(f"{Fore.CYAN}Date of Analysis: {start_date} {Style.RESET_ALL}")
    print("-" * 50)
    print(prediction_statement)
    print("=" * 50)

    # Call the summarize API
    print("\nSTOCK NEWS SUMMARY ON {} \n".format(start_date))
    my_list = news_data['title'][0]
    s = ';'.join(my_list)
    result = summarize_stock_news(s)
    print(result)

else:
    print("Take A Break!!!! Stock Market is CLOSED TODAY")


# Call the function to display the plot
plot_stock_data()