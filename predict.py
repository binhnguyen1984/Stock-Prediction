from sklearn.model_selection import train_test_split

import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from data_utils import load_data, create_dataset, reshape_data
from train import train_lstm
from sklearn.preprocessing import MinMaxScaler


def plot_results_multiple(predicted_data, true_data):
    plt.plot(np.array(predicted_data).reshape(-1,1)[:,0], label="Predictions")
    plt.plot(np.array(true_data).reshape(-1,1)[:,0], label="True data")
    plt.legend()
    plt.show()

def predict_sequences_full(model, curr_frame, prediction_length, scaler):
    """
    Make a sequence of prediction_length predictions starting from the input sequence
    """
    timesteps = curr_frame.shape[0]
    predicted = []
    for _ in range(prediction_length):
        predicted.append(scaler.inverse_transform(model.predict(curr_frame[newaxis,:,:]))[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, timesteps-1, predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, timesteps, prediction_len, scaler):
    """
    predict a sequence of prediction_len predictions and then shifting the input sequence to prediction_len steps forward
    """
    prediction_seqs = []
    for i in range(len(data)//prediction_len):
        curr_frame = data[i*prediction_len]
        predicted = []
        for _ in range(prediction_len):
            prediction = scaler.inverse_transform(model.predict(curr_frame[newaxis,:,:]))[0,0]
            predicted.append(prediction)
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, timesteps-1, predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


stock_prices = load_data(filename="HistoricalQuotes.csv")
scaler = MinMaxScaler(feature_range=(0, 1))
stock_prices = scaler.fit_transform(stock_prices)

train, test = train_test_split(stock_prices, test_size = 0.2, random_state = 0)

timesteps = 5
X_train, y_train = create_dataset(train, look_back=timesteps)
X_train = reshape_data(X_train)

model = train_lstm(X_train, y_train)
X_test, y_test = create_dataset(test, look_back=timesteps)
X_test = reshape_data(X_test)

prediction_length = len(y_test)
predictions = predict_sequences_full(model, X_test[0], prediction_length, scaler)
y_test = scaler.inverse_transform(y_test)
plot_results_multiple(predictions, y_test)
