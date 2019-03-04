from keras.callbacks import ModelCheckpoint
from model import build_model


def train_lstm(X_train, y_train, saved_model_name="lstm_model.h5"):
    checkpointer = ModelCheckpoint(filepath=saved_model_name, verbose=1, save_best_only=True)
    model = build_model([128,128], dropouts=[0.5, 0.3], timesteps = X_train.shape[1], features=1)
    model.fit(X_train, y_train, batch_size=4, epochs=1000, callbacks=[checkpointer])
    return model

