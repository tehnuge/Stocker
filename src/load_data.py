from yahoo_fin import stock_info as si
from sklearn import preprocessing
import pandas as pd
import numpy as np
from collections import deque
from sklearn.model_selection import train_test_split
import math

def SMA(x, period, df):
    if x - period < 0:
        return
    avgSum = 0
    for y in range(0, period):
        # column 3 is closing price
        avgSum = avgSum + df.iat[x - y, 3]
    
    avg = avgSum /period
    return avg

def SD(x, period, df, avg):
    if x - period < 0:
        return
    sum = 0
    for y in range(0, period):
        sum = sum + (df.iat[x - y, 3] - avg)**2
    sd = math.sqrt(sum / period)
    return sd    
    
def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, 
                test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low', "sma50", "sma200"]):
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
        # add columns (starting at 7)
        df['sma50'] = 0
        df['sma200'] = 0
        df['+bol20'] = 0
        df['-bol20'] = 0

        # iterate through table and set SMA cell value
        for x in range(0, len(df['close'])):
            df.iat[x, 7] = SMA(x, 50, df)
            df.iat[x, 8] = SMA(x, 200, df)
            # Bollinger's band calculations
            sma20 = SMA(x, 20, df)
            sd = SD(x, 20, df, sma20)
            if sd is not None and sma20 is not None:
                df.iat[x, 9] = sma20 + 2 * sd
                df.iat[x, 10] = sma20 - 2 * sd
            
        # drop NaN
        df.dropna(inplace=True)
        print(df)

    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker

    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()

    # return actual stock price
    result['actual'] = df.tail(1)['close'].array[0].item()

    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
    # this last_sequence will be used to predict in future dates that are not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    # shift the last sequence by -1
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    # split the dataset
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    # return the result
    return result
