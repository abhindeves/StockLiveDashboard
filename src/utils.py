import pandas as pd
from dotenv import find_dotenv,load_dotenv
import os
from openai import OpenAI
import logging  # Import logging module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=OPENAI_API_KEY)


def data_combiner(stock_price,stock_news):
    stock_price.set_index('index',inplace=True)
    stock_news.set_index('Date',inplace=True)
    
    final_price_news = stock_price.join(stock_news)
    logging.info("Data combined successfully.")
    return final_price_news

def prepare_final_data(df):

    X = df[['Open', 'High', 'Close', 'Subjectivity', 'Polarity', 'compound','neg', 'pos', 'neu', 'MA7', 
                     'MA20', 'MACD', '20SD', 'upper_band','lower_band', 'EMA', 'logmomentum']]

    logging.info("Preparing final data for model input.")
    return X

def append_new_data_to_existing(todays_data):
    file_path = "data\MASTER_MSFT_FINAL.csv"
    master_data = pd.read_csv(file_path, index_col=0).tail(2)
    todays_data = todays_data.reset_index().rename(columns={'index':'Date'})
    todays_data['Date'] = todays_data['Date'].astype('str')
    
    todays_data_date = str(todays_data['Date'].iloc[0])
    logging.info(f"Checking if today's data date {todays_data_date} exists in master data.")

    if todays_data_date.split()[0] not in  master_data['Date'].astype(str).to_list():
        logging.info("Today's data is new, appending to master data.")
        master_data = master_data._append(todays_data,ignore_index=True)
        print(master_data)
        master_data.to_csv('data\MASTER_MSFT_FINAL.csv')
        logging.info("Data put into MASTER DATA")
    else:
        logging.info("Today's data already exists in master data, skipping append.")

    
    

def summarize(s) ->str:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=[
            {"role": "system", "content": "You are a highly skilled AI trained in Stock Market News Summarization. I would like you to read the list of News headlines and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the overall market trend without needing to read the entire text. You can start by greeting the Day trader in an appealing manner. Please avoid unnecessary details or tangential points."},
            {"role": "user", "content": s}
        ]
    )

    logging.info("Generating summary for the provided news headlines.")
    return completion.choices[0].message.content

def print_pred(pred,start_date,end_date)->None:
    prediction_statement = ""
    if pred == 1:
        prediction_statement = f"The stock price is predicted to go UP on {end_date}."
    elif pred == 0:
        prediction_statement = f"The stock price is predicted to go DOWN {end_date}."
    
    logging.info(f"Prediction made for date {end_date}: {prediction_statement}")
    print("=" * 50)
    print(f"📈📉 Stock Price Prediction 📈📉")
    print("=" * 50)
    print(f"Date of Analysis: {start_date}")
    print("-" * 50)
    print(prediction_statement)
    print("=" * 50)
