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
tickers = ["GOOG", "NFLX", "AAPL", "AMZN", "GM", "COST", "INCAF", "FNV", "CEF"]
# tickers = ["INCAF", "FNV", "CEF"]

# Window size or the sequence length
N_STEPS = 100
# Lookup step, 1 is the next day
LOOKUP_STEP = 1
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low", "sma50", "sma200", "+bol20","-bol20"]
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

      
def train_model(ticker, n_steps, lookup_step, test_size, feature_columns, loss, units, cell, n_layers, dropout, optimizer, bidirectional):
  create_folders()
  ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
  # model name to save, making it as unique as possible based on parameters
  model_name = f"{date_now}_{ticker}-{loss}-{optimizer}-{cell.__name__}-seq-{n_steps}-step-{lookup_step}-layers-{n_layers}-units-{units}"
  if bidirectional:
      model_name += "-b"
  # set seed, so we can get the same results after rerunning several times
  np.random.seed(314)
  tf.random.set_seed(314)
  random.seed(314)
  # load the data
  data = load_data(ticker, n_steps, lookup_step=lookup_step, test_size=test_size, feature_columns=feature_columns)

  # save the dataframe
  data["df"].to_csv(ticker_data_filename)

  # construct the model
  model = create_model(n_steps, loss=loss, units=units, cell=cell, n_layers=n_layers,
                      dropout=dropout, optimizer=optimizer, bidirectional=bidirectional)

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

def test_model(ticker, n_steps, lookup_step, test_size, feature_columns, loss, units, cell, n_layers, dropout, optimizer, bidirectional):
  ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
  # model name to save, making it as unique as possible based on parameters
  model_name = f"{date_now}_{ticker}-{loss}-{optimizer}-{cell.__name__}-seq-{n_steps}-step-{lookup_step}-layers-{n_layers}-units-{units}"
  if bidirectional:
      model_name += "-b"
  data = load_data(ticker, n_steps, lookup_step=lookup_step, test_size=test_size,
                feature_columns=feature_columns, shuffle=False)

  # construct the model
  model = create_model(n_steps, loss=loss, units=units, cell=cell, n_layers=n_layers,
                      dropout=dropout, optimizer=optimizer, bidirectional=bidirectional)

  model_path = os.path.join("results", model_name) + ".h5"
  model.load_weights(model_path)


  # evaluate the model
  mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
  # convert mae to np.float32; changed from py 3.7
  mae = np.array([mae])[0]
    # calculate the mean absolute error (inverse scaling)
  mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform(mae.reshape(1, -1))[0][0]
  print("Mean Absolute Error:", mean_absolute_error)
  # predict the future price
  future_price = predict(model, data, n_steps)
  print(f"Future price after {lookup_step} days is {future_price:.2f}$")
  accuracy = get_accuracy(model, data, lookup_step)
  logData(date_now, ticker, mean_absolute_error, future_price, data['actual'], EPOCHS, lookup_step, accuracy)
  # plot_graph(model, data)



# if tf.test.gpu_device_name():
# 	print('gpu is{}'.format(tf.test.gpu_device_name()))
# else:
# 	print('no gpu')
for ticker in tickers:
  train_model(ticker, N_STEPS, LOOKUP_STEP, TEST_SIZE, FEATURE_COLUMNS, LOSS, UNITS, CELL, N_LAYERS, DROPOUT, OPTIMIZER, BIDIRECTIONAL)
  test_model(ticker, N_STEPS, LOOKUP_STEP, TEST_SIZE, FEATURE_COLUMNS, LOSS, UNITS, CELL, N_LAYERS, DROPOUT, OPTIMIZER, BIDIRECTIONAL)

