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

def predict_sequences_multiple(model, data, timesteps, scaler):
    """
    predict a sequence of prediction_len predictions and then shifting the input sequence to prediction_len steps forward
    """
    prediction_seqs = []
    for i in range(len(data)-timesteps):
        curr_frame = data[i:i+timesteps,:]
        prediction = scaler.inverse_transform(model.predict(curr_frame[newaxis,:,:]))[0,0]
        prediction_seqs.append(prediction)
    return prediction_seqs


stock_prices = load_data(filename="HistoricalQuotes.csv")

train, test = train_test_split(stock_prices, test_size = 0.2, random_state = 0)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)


timesteps = 5
X_train, y_train = create_dataset(train_scaled, look_back=timesteps)
X_train = reshape_data(X_train)

model = train_lstm(X_train, y_train)
data_total = np.concatenate([train, test])
inputs = data_total[len(data_total)-len(test)-timesteps:]
inputs = scaler.transform(inputs)
inputs = inputs.reshape((-1,1))
predictions = predict_sequences_multiple(model, inputs, timesteps, scaler)
plot_results_multiple(predictions, test)
