import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
from load_data import load_data
from util import *

# # TODO: from https://rustyonrampage.github.io/deep-learning/2018/10/18/tensorfow-amd.html make GPU work
# import plaidml.keras
# plaidml.keras.install_backend()

# import keras
# from keras import backend as K
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random

# Apple stock market
ticker = "AAPL"

# Window size or the sequence length
N_STEPS = 100
# Lookup step, 1 is the next day
LOOKUP_STEP = 5
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")
### model parameters
N_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False
### training parameters
# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 400

ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"
    

def create_model(sequence_length, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), input_shape=(None, sequence_length)))
            else:
                model.add(cell(units, return_sequences=True, input_shape=(None, sequence_length)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model

# Main program
def create_folders():
  if not os.path.isdir("results"):
      os.mkdir("results")
  if not os.path.isdir("logs"):
      os.mkdir("logs")
  if not os.path.isdir("data"):
      os.mkdir("data")
      
def train_model():
  create_folders()
  # set seed, so we can get the same results after rerunning several times
  np.random.seed(314)
  tf.random.set_seed(314)
  random.seed(314)
  # load the data
  data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)

  # save the dataframe
  data["df"].to_csv(ticker_data_filename)

  # construct the model
  model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                      dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

  # some tensorflow callbacks
  checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
  tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

  history = model.fit(data["X_train"], data["y_train"],
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_data=(data["X_test"], data["y_test"]),
                      callbacks=[checkpointer, tensorboard],
                      verbose=1)

  model.save(os.path.join("results", model_name) + ".h5")

def test_model():
  data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                feature_columns=FEATURE_COLUMNS, shuffle=False)

  # construct the model
  model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                      dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

  model_path = os.path.join("results", model_name) + ".h5"
  model.load_weights(model_path)


  # evaluate the model
  mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
  mae = np.array([mae])[0]
  print(type(mae))
    # calculate the mean absolute error (inverse scaling)
  mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform(mae.reshape(1, -1))[0][0]
  print("Mean Absolute Error:", mean_absolute_error)
  # predict the future price
  future_price = predict(model, data)
  print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
  accuracy = get_accuracy(model, data)
  logData(date_now, ticker, mae, future_price, '0', EPOCHS, LOOKUP_STEP, accuracy)
  # plot_graph(model, data)



# if tf.test.gpu_device_name():
# 	print('gpu is{}'.format(tf.test.gpu_device_name()))
# else:
# 	print('no gpu')

# train_model()
test_model()

