from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout, Activation

def build_model(neurons, dropouts, timesteps =1, features=1):
    model = Sequential()
    model.add(LSTM(neurons[0], input_shape=(timesteps, features), return_sequences=True))
    model.add(Dropout(dropouts[0]))
    for i in range(len(neurons)-1):
        model.add(LSTM(neurons[i], return_sequences=True))
        model.add(Dropout(dropouts[i]))
        
    model.add(LSTM(neurons[-1], return_sequences = False))
    model.add(Dropout(dropouts[-1]))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(optimizer="rmsprop", loss="mse")
    return model

