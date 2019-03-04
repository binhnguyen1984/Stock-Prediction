import pandas as pd
import numpy as np

def create_dataset(data, look_back=1):
    """
    create two sets of data: X contains data from a certain time steps in the past
    while Y contains the data at one time step in the future
    """
    X = []
    Y = []
    for i in range(len(data)-look_back):
        X.append(data[i:i+look_back])
        Y.append(data[i+look_back])
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

# reshape the data to obtain [samples, timestep, features]
def reshape_data(data):
    data = np.reshape(data, (data.shape[0], data.shape[1],1))
    return data

def load_data(filename):
    df = pd.read_csv(filename)    
    stock_prices = np.array(df.iloc[:,0]).reshape(-1,1)
    return stock_prices

