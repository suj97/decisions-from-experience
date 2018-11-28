"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Masking, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# transform list into supervised learning format
def series_to_supervised(data, n_in, n_out=1):
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	agg.dropna(inplace=True)
	return agg.values

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# difference dataset
def difference(data, order):
	return [data[i] - data[i - order] for i in range(order, len(data))]


# forecast with the fit model
def model_predict(model, history, network):
    # unpack config
    n_input = network['n_input']
    n_diff = network['n_diff']
    # prepare data
    correction = 0.0
    if n_diff > 0:
        correction = history[-n_diff]
        history = difference(history, n_diff)
    # shape input for model
    x_input = array(history[-n_input:]).reshape((1, n_input))
    # make forecast
    yhat = model.predict(x_input, verbose=0)
    # correct forecast if it was differenced
    return correction + yhat[0]



def model_compile(train, network):
	# unpack config
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']
    n_input = network['n_input']
    n_epochs = network['n_epochs']
    n_batch = network['n_batch']
    n_diff = network['n_diff']
    
	# prepare data
    X_train, y_train, X_test, y_test = train
 #    if n_diff > 0:
 #        train = difference(train, n_diff)
	# # transform series into supervised format
 #    data = series_to_supervised(train, n_in=n_input)
	# # separate inputs and outputs
 #    train_x, train_y = data[:, :-1], data[:, -1]
	# define model
    # model = Sequential()

    model = Sequential()
    
    model.add(Masking(mask_value=-100.0, input_shape=(None, X_train.shape[2])))
    
    if(nb_layers == 1):
        model.add(LSTM(nb_neurons, activation=activation, input_shape=(None, X_train.shape[2])))
        model.add(Dropout(0.2))
    else:
        model.add(LSTM(nb_neurons, activation=activation, input_shape=(None, X_train.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        
        while(nb_layers-2):
            model.add(LSTM(nb_neurons, activation=activation, return_sequences=True))
            model.add(Dropout(0.2))
            nb_layers -= 1
        model.add(LSTM(nb_neurons, activation=activation))
        model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    
    # model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch)
   
    
    #     # Add each layer.
    # for i in range(nb_layers):

    #     # Need input shape for first layer.
        
    #     if i == 0:
    #         model.add(Dense(nb_neurons, activation=activation, input_dim=n_input))
    #     else:
    #         model.add(Dense(nb_neurons, activation=activation))
            

    #     model.add(Dropout(0.2))  # hard-coded dropout

    # # Output layer.
    # model.add(Dense(1, activation='softmax'))
    

    # model.compile(loss='mse', optimizer=optimizer,
    #               metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model

################################ my functions ########################################################

def parse_dataset(name):
    infile = open(name, 'r')
    lines = infile.readlines()
    data = []
    for line in lines:
        data.append([float(x) for x in line.split(',')])
    y = [x[-1] for x in data]
    data = [x[:-1] for x in data]
    return data, y


################################ my functions ########################################################


def train_and_score(network, dataset, n_test):
    
    # series = read_csv(dataset, header=0, index_col=0)
    # data = series.values
    X_train, y_train = parse_dataset('../data/estimation_without_padding.csv')
    X_test, y_test = parse_dataset('../data/competition_without_padding.csv')
    maxlen1 = max([len(x) for x in X_train])
    maxlen2 = max([len(x) for x in X_test])
    X_train = pad_sequences(X_train, padding='post', value=-100.0, dtype=float, maxlen=maxlen1)
    X_test = pad_sequences(X_test, padding='post', value=-100.0, dtype=float, maxlen=maxlen2)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]//2, 2))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]//2, 2))

    # data split
    predictions = list()
    
    # split dataset
    # train, test = train_test_split(data, n_test)
    train = (X_train, y_train, X_test, y_test)
    
    model = model_compile(train, network)
    
 #    history = [x for x in train]
	# # step over each time-step in the test set
 #    for i in range(len(test)):
	# 	# fit model and make forecast for history
 #        yhat = model_predict(model, history, network)
	# 	# store forecast in list of predictions
 #        predictions.append(yhat)
	# 	# add actual observation to history for the next loop
 #        history.append(test[i])
	# estimate prediction error
    scores = model.evaluate(X_test, y_test, verbose=0)
    error = 100.0 - scores[1]*100
    # error = measure_rmse(test, predictions)
    print(' > %.3f' % error)
    return error
