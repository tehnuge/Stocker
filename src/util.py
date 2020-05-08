from sklearn.metrics import accuracy_score
N_STEPS = 100
LOOKUP_STEP = 5
import numpy as np
import matplotlib.pyplot as plt
import csv


def predict(model, data, classification=False):
  # retrieve the last sequence from data
  last_sequence = data["last_sequence"][:N_STEPS]
  # retrieve the column scalers
  column_scaler = data["column_scaler"]
  # reshape the last sequence
  last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
  # expand dimension
  last_sequence = np.expand_dims(last_sequence, axis=0)
  # get the prediction (scaled from 0 to 1)
  prediction = model.predict(last_sequence)
  # get the price (by inverting the scaling)
  predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[0][0]
  return predicted_price
  
def plot_graph(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    # last 200 days, feel free to edit that
    plt.plot(y_test[-200:], c='b')
    plt.plot(y_pred[-200:], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

def get_accuracy(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
    return accuracy_score(y_test, y_pred)

def logData(date, ticker, mae, predicted, actual, epochs, lookup_step, accuracy):
  mydict = [{'date': date, 'ticker': ticker, 'mae': mae, 'predicted': predicted, 'actual': actual, 'epochs': epochs, 'lookup_step': lookup_step, 'accuracy': accuracy} ]
          # field names  
  fields = ['date', 'ticker', 'mae', 'predicted', 'actual', 'epochs', 'lookup_step', 'accuracy'] 
  filename = 'data_logs.csv' 
  # writing to csv file  
  with open(filename, 'a') as csvfile:  
      # creating a csv dict writer object  
      writer = csv.DictWriter(csvfile, fieldnames = fields)  
          
      # writing headers (field names)  
      #writer.writeheader()  
          
      # writing data rows  
      writer.writerows(mydict)  