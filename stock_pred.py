import pandas as pd
import pandas_datareader as web
from pandas_datareader import data as pdr
import yfinance as yf
import keras_tuner as HyperResNet

from tensorflow import keras as kr
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import math
import matplotlib.pyplot as plt

"""Splits historical stock data into features and classification.

XY_split splits data into feature data and classification data on yfinance stock data for LSTM and/or ML prediction analyses. Classification data is a given stock price whereas feature data is the stock prices from the past n days. 

  Typical usage example:

  XY_split(data, number of days)
  test_x, test_y = XY_split(test_data, 30)
"""
def XY_split(data, n):
  x = [] #independent, contains the past N closing values (using 60 for this model)
  y = [] #feature

  for i in range(n, len(data)):
    x.append(data[i-n:i,0])
    y.append(data[i,0])
  
  x = np.array(x)
  y = np.array(y)
  
  return x, y

"""Reshapes 2 dimensional data to 3 dimensions

reshaping_dataset reshapes two dimensional data to three dimensions

  Typical usage example:

  reshaping_dataset(data)
  3d_data = reshaping_dataset(2d_data)
"""
def reshaping_dataset(data):
  x = np.reshape(data, (data.shape[0],data.shape[1],1))
  return x

'''
  5 Layer Long Short-Term Memory Model

  Layer 1: 128 units LSTM 
  Layer 2: 64 units LSTM
  Layer 3: 64 units LSTM
  Layer 4: 25 units Dense (Regular Neural Network)
  Layer 5: 1 unit Dense (Return Layer)

  Optimized to minimize MSE
'''
def model_creation(hp):
  model = kr.models.Sequential()
  model.add(kr.layers.LSTM(128, return_sequences=True, input_shape=(training_x.shape[1], training_x.shape[2])))
  model.add(kr.layers.LSTM(64, return_sequences=True))
  model.add(kr.layers.LSTM(64, return_sequences=False))
  model.add(kr.layers.Dense(25))
  model.add(kr.layers.Dense(1))
  model.compile(optimizer='adam', loss='mean_squared_error') 

"""Returns tomorrow's predicted stock prices

predict uses an LSTM ML model to predict tomorrows stock data given a ticker and a date. The ticker indicates the stock to be predicted and the date reflects the beginning of data collection. 

  Typical usage example:

  predict(tkr, date)
  predict("AAPL", "2012-01-01")
"""
def predict(tkr, date, hyperparametization = "off"):
  yf.pdr_override()
  scaler = MinMaxScaler(feature_range=(0,1))

  ### GETTING DATA ###

  stock_name = tkr

  #test daate: 2012-01-01
  tkr = pdr.get_data_yahoo(tkr, start=date)

  print()
  print(f'{stock_name} Data Accessed || Starting on {date}')
  #initial plot// you need to figure out how to save this onto the person's computer

  plt.figure(figsize=(8,4))
  plt.title('Close Price History')
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Close Price US', fontsize=18)
  plt.plot(tkr['Close'])
  plt.savefig('hist.jpg')


  ### DATA PROCESSING ###

  prices_quote = tkr['Close'].to_numpy()
  prices = prices_quote.reshape(-1,1) #reshaping the array for LSTM input
  lenPrices = len(prices)

  #SCALING the data to fit between (0 and 1)
  scaled_data = scaler.fit_transform(prices)
  scaled_data

  #SPLITTING the data into training and testing (6:4)
  train, test = np.split(scaled_data,[int(0.6*lenPrices)])

  train_len = len(train)
  train_len = int(train_len)

  #SPLITTING the datasets into features and close
  training_x, training_y = XY_split(train, 60)

  #CORRECT testing dataset to reflect non-scaled data
  testing_x, testing_y = XY_split(test, 60)
  testing_y = prices[train_len+60:, :]

  training_x = reshaping_dataset(training_x)
  testing_x = reshaping_dataset(testing_x)

  print()
  print(f'{stock_name} Data Processing Finished!')
  
  
  ## LSTM GENERATION ##

  '''
  5 Layer Long Short-Term Memory Model

  Layer 1: 128 units LSTM 
  Layer 2: 64 units LSTM
  Layer 3: 64 units LSTM
  Layer 4: 25 units Dense (Regular Neural Network)
  Layer 5: 1 unit Dense (Return Layer)

  Optimized to minimize MSE
  '''
  model = kr.models.Sequential()
  model.add(kr.layers.LSTM(128, return_sequences=True, input_shape=(training_x.shape[1], training_x.shape[2])))
  model.add(kr.layers.LSTM(64, return_sequences=True))
  model.add(kr.layers.LSTM(64, return_sequences=False))
  model.add(kr.layers.Dense(25))
  model.add(kr.layers.Dense(1))
  model.compile(optimizer='adam', loss='mean_squared_error') 

  print()
  print('LSTM Model Generated')

  #train the model
  model.fit(training_x, training_y, batch_size = 1, epochs = 1)

  ## TESTING MODEL##
  predictions = model.predict(testing_x)
  predictions = scaler.inverse_transform(predictions)

  #Get the root mean squared error (RMSE)
  rmse = np.sqrt(np.mean(predictions - testing_y)**2)

  print()
  print(f'RMSE on testing data: {rmse}')
  
  ## visualizing results
  train = prices[:train_len]
  valid = prices[train_len + 60:]

  predictTomorrow = predictions[-1]
  
  print(f'prediction for tmr: {predictTomorrow}')


  valid = pd.DataFrame(valid)
  valid = valid.set_index(tkr.index[lenPrices-len(valid):])
  valid = valid.rename(columns = {0:'Close'})

  train = pd.DataFrame(train)
  train = train.set_index(tkr.index[:train_len])
  train = train.rename(columns = {0:'Close'})

'''
  #Plot the data
  valid["Predictions"] = predictions

  #visualize the data
  plt.figure(figsize=(16,8))
  plt.title('Model Accuracy')
  plt.xlabel("Date")
  plt.ylabel("Close Price")

  plt.plot(train['Close'])
  plt.plot(valid[['Close', 'Predictions']])
  plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
  plt.savefig('prediction.jpg')

  print(valid)
'''