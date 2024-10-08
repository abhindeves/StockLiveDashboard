import pandas as pd

def data_combiner(stock_price,stock_news):
    stock_price.set_index('index',inplace=True)
    stock_news.set_index('Date',inplace=True)
    
    final_price_news = stock_price.join(stock_news)
    return final_price_news

def prepare_final_data(df):

    X = df[['Open', 'High', 'Close', 'Subjectivity', 'Polarity', 'compound','neg', 'pos', 'neu', 'MA7', 
                     'MA20', 'MACD', '20SD', 'upper_band','lower_band', 'EMA', 'logmomentum']]

    return X

def append_new_data_to_existing(todays_data):
    file_path = "data\MASTER_MSFT_FINAL.csv"
    master_data = pd.read_csv(file_path, index_col=0).tail(2)
    todays_data = todays_data.reset_index().rename(columns={'index':'Date'})
    todays_data['Date'] = todays_data['Date'].astype('str')
    
    todays_data_date = str(todays_data['Date'].iloc[0])
    if todays_data_date.split()[0] != master_data['Date'].iloc[-1]:
        # master_data = pd.concat([master_data, todays_data], ignore_index=True)
        master_data = master_data._append(todays_data,ignore_index=True)
        print(master_data)
    else:
        pass
    print(master_data)
    

def print_prediction(prediction):
    pass
