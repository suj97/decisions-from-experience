import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Embedding, Conv1D, MaxPooling1D
from keras import optimizers,regularizers
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder
from sklearn_pandas import DataFrameMapper
from keras.preprocessing.sequence import pad_sequences

dataset = pd.read_csv('data/estimation.csv')
dataset_comp = pd.read_csv('data/competition.csv')

print(dataset['1'].shape)
norm = [str(i) for i in xrange(1, 432, 2)]
binar = [str(i) for i in xrange(0, 432, 2)]
#print(dataset['0'])

y_train = dataset.iloc[:, -1].values
y_test = dataset_comp.iloc[:, -1].values

mapper = DataFrameMapper([(norm, StandardScaler()), (binar, OneHotEncoder(sparse=False))])
dataset = mapper.fit_transform(dataset)
dataset_comp = mapper.fit_transform(dataset_comp)
print(dataset.shape)
print(dataset_comp.shape)

#X_train = dataset.iloc[:, 0:-1].values
#y_train = dataset.iloc[:, -1].values
#X_test = dataset_comp.iloc[:, 0:-1].values
#y_test = dataset_comp.iloc[:, -1].values

X_train = dataset
#y_train = dataset[:, -1]
X_test = dataset_comp
#y_test = dataset_comp[:, -1]
print(y_train)

input_length = len(X_train[0])
X_test = pad_sequences(X_test, padding='post', maxlen=input_length)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# X_train, y_train = X, y

y_train -= 1
y_test -= 1

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
#print(X_train[0])

no_units = [1,3,5,10,30,50,100,200,300,500]
#no_units=[5]
d = 0.2

#outfile = open('diff-no-units.txt', 'w')

for units in no_units:
    # create the model
    model = Sequential()
    # model.add(Embedding(input_dim = 3, output_dim = 2, input_length = input_length))
    model.add(LSTM(units, return_sequences=False, activation='relu',input_shape=(1, input_length)))
    #model.add(Dropout(d))
    #model.add(LSTM(units, return_sequences=False, activation='relu'))
    #model.add(Dropout(d))
    #model.add(LSTM(units, activation='relu', return_sequences=True))
    #model.add(Dropout(d))
    #model.add(LSTM(units, activation='relu', return_sequences=True))
    #model.add(Dropout(d))
    #model.add(LSTM(units, activation='relu', return_sequences=True))
    #model.add(Dropout(d))
    #model.add(LSTM(units, activation='relu', return_sequences=False))
    #model.add(Dropout(d))
    model.add(Dense(1, activation='sigmoid'))

    opt = optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=5)
    
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    scores1 = model.evaluate(X_train, y_train, verbose=0)
    print("Accuracy: %.2f%%" % (scores1[1]*100))
    #outfile.write(str((units, scores[1]*100, scores1[1]*100)))
#outfile.close()
