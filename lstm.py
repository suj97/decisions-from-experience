import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Embedding
from keras import optimizers
from keras.regularizers import L1L2 
data_without_outcomes = 0

if(data_without_outcomes):
    dataset = pd.read_csv('data/estimation_without_outcomes.csv')
    dataset_comp = pd.read_csv('data/competition_without_outcomes.csv')
else:
    dataset = pd.read_csv('data/estimation.csv')
    dataset_comp = pd.read_csv('data/competition.csv')
    
X_train = dataset.iloc[:, 0:-1].values
y_train = dataset.iloc[:, -1].values
X_test = dataset_comp.iloc[:, 0:-1].values
y_test = dataset_comp.iloc[:, -1].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# X_train, y_train = X, y

input_length = len(X_train[0])

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

res=[]
#lstm_units = [10,30,50,100,300,500]
#no_epochs = [10,20,30,50,100,200,300,500]
#no_epochs=[20]
layers=1
#batch_sizes= [2,5,10,20,30,50,80,100]
batch_sizes = [20]
reg=L1L2(l1=0.00, l2=0.01)
for batch_size in batch_sizes:
    # create the model
    model = Sequential()
    # model.add(Embedding(input_dim = 3, output_dim = 2, input_length = input_length))
    if(layers!=1):
        model.add(LSTM(30, return_sequences=True, input_shape=(1, input_length)))
    else:
        model.add(LSTM(30, bias_regularizer=reg, kernel_regularizer=reg, input_shape=(1, input_length)))
    l = 1
    while(l+1<layers):
        model.add(LSTM(30, return_sequences=True))
        l+=1
    if(layers>1):
        model.add(LSTM(30))
    model.add(Dense(1, kernel_regularizer=reg, bias_regularizer=reg))
    
    opt = optimizers.Adam()
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=batch_size, verbose=0)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    scores1 = model.evaluate(X_train, y_train, verbose=0)
    print("Accuracy: %.2f%%" % (scores1[1]*100))
    res.append((batch_size, round(scores[1]*100, 2), round(scores1[1]*100, 2)))
#outfile = open("diff-no-bs.txt",'w')
#outfile.write(str(res))
#outfile.close()
