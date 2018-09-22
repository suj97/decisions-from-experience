import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Embedding
from keras import optimizers
from matplotlib import pyplot

units=10
loss='binary_crossentropy'
epochs=500
batch_size=20

dataset = pd.read_csv('data/estimation.csv')
dataset_comp = pd.read_csv('data/competition.csv')

X_train = dataset.iloc[:, 0:-1].values
y_train = dataset.iloc[:, -1].values
X_test = dataset_comp.iloc[:, 0:-1].values
y_test = dataset_comp.iloc[:, -1].values

y_train -= 1
y_test -= 1

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# X_train, y_train = X, y

input_length = len(X_train[0])

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# create the model
model = Sequential()
# model.add(Embedding(input_dim = 3, output_dim = 2, input_length = input_length))
model.add(LSTM(units, input_shape=(1, input_length)))
model.add(Dense(1, activation='sigmoid'))

opt = optimizers.Adam()
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()
pyplot.savefig('x.png')

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
scores1 = model.evaluate(X_train, y_train, verbose=0)
print("Accuracy: %.2f%%" % (scores1[1]*100))
