import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Embedding
from keras import optimizers

dataset = pd.read_csv('data/ndata.csv')

X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X_train, y_train = X, y

input_length = len(X_train[0])

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# create the model
model = Sequential()
# model.add(Embedding(input_dim = 3, output_dim = 2, input_length = input_length))
model.add(LSTM(100, input_shape=(1, input_length)))
model.add(Dense(1))

opt = optimizers.Adam()
model.compile(loss='mae', optimizer=opt, metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=300, batch_size=10)

# Final evaluation of the model
scores = model.evaluate(X_train, y_train, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
