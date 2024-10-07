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
    # Add todays data to the table in order to plot graphs
    # Read master data
    file_path = "data\MASTER_MSFT_FINAL.csv"
    master_data = pd.read_csv(file_path, index_col=0).tail(2)
    todays_data = todays_data.reset_index().rename(columns={'index':'Date'})
    print(master_data)
    print(todays_data)
    # Check if the date from merged dataset is already in master data
    # merged_date = pd.to_datetime(todays_data['Date'])
    # print(type(list(master_data['Date'].values)))
    # print(type(str(merged_date)))
    # if str(merged_date) not in list(master_data['Date'].values):
    #     print("merged data not in the list")
    #     # If date is not present, concatenate the merged data
    #     master_data = pd.concat([master_data, todays_data.reset_index()], ignore_index=True)
    #     print(master_data)
    # else:
    #     pass
