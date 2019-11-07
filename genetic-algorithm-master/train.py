"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error

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
    if n_diff > 0:
        train = difference(train, n_diff)
	# transform series into supervised format
    data = series_to_supervised(train, n_in=n_input)
	# separate inputs and outputs
    train_x, train_y = data[:, :-1], data[:, -1]
	# define model
    model = Sequential()
   
    
        # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_dim=n_input))
        else:
            model.add(Dense(nb_neurons, activation=activation))
            

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(1, activation='softmax'))
    

    model.compile(loss='mse', optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    return model


def train_and_score(network, dataset, n_test):
    
    series = read_csv(dataset, header=0, index_col=0)
    data = series.values
    # data split
    predictions = list()
    
    # split dataset
    train, test = train_test_split(data, n_test)
    
    model = model_compile(train, network)
    
    history = [x for x in train]
	# step over each time-step in the test set
    for i in range(len(test)):
		# fit model and make forecast for history
        yhat = model_predict(model, history, network)
		# store forecast in list of predictions
        predictions.append(yhat)
		# add actual observation to history for the next loop
        history.append(test[i])
	# estimate prediction error
    error = measure_rmse(test, predictions)
    print(' > %.3f' % error)
    return error
