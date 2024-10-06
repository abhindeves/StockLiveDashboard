

def data_combiner(stock_price,stock_news):
    stock_price.set_index('index',inplace=True)
    stock_news.set_index('Date',inplace=True)
    
    final_price_news = stock_price.join(stock_news)
    return final_price_news

def prepare_final_data(df):

    X = df[['Open', 'High', 'Close', 'Subjectivity', 'Polarity', 'compound','neg', 'pos', 'neu', 'MA7', 
                     'MA20', 'MACD', '20SD', 'upper_band','lower_band', 'EMA', 'logmomentum']]

    return X