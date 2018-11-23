import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, LSTM, GRU, SimpleRNN, Embedding, Conv1D, MaxPooling1D, Masking
from keras import optimizers,regularizers
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

def create_dataset_train(data, n_steps):
    X_train, Y_train= list(), list()
    final_choices = [int(x[-1]) for x in data]
    features = [x[:-1] for x in data]
    # making training data 
    count = 0
    for feat,f_choice in zip(features, final_choices):
        if(len(feat)//2 <= n_steps):
            X_train.append(feat)
            Y_train.append(f_choice)
            count += 1
        else:
            for i in range(len(feat)//2-n_steps):
                end_idx = 2*(i+n_steps)
                X_train.append(feat[2*i:end_idx])
                Y_train.append(feat[end_idx])
            X_train.append(feat[(len(feat)-2*n_steps):len(feat)])
            Y_train.append(f_choice)
#     print(count)
    return X_train, Y_train


def create_dataset_test(data, n_steps):
    X_test, Y_test= list(), list()
    final_choices = [int(x[-1]) for x in data]
    features = [x[:-1] for x in data]
    # making training data 
    count = 0
    for feat,f_choice in zip(features, final_choices):
        if(len(feat)//2 <= n_steps):
            X_test.append(feat)
            Y_test.append(f_choice)
            count += 1
        else:
            X_test.append(feat[(len(feat)-2*n_steps):len(feat)])
            Y_test.append(f_choice)
#     print(count)
    return X_test, Y_test

def parse_dataset(name, n_steps, flag):
    infile = open(name, 'r')
    lines = infile.readlines()
    data = []
    for line in lines:
        data.append([float(x) for x in line.split(',')])
#     normalise_outcomes(data)
    if(flag):
        return create_dataset_train(data, n_steps)
    else:
        return create_dataset_test(data, n_steps)

def get_dataset(n_steps):
    X_train, y_train = parse_dataset('data/estimation_without_padding.csv',n_steps, 1)
    X_test, y_test = parse_dataset('data/competition_without_padding.csv',n_steps, 0)
    
    maxlen = max(max([len(x) for x in X_train]), max([len(x) for x in X_test]))
    
    X_train = pad_sequences(X_train, padding='post', value=-100, dtype=float, maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', value=-100, dtype=float, maxlen=maxlen)
    
    X_train = np.reshape(X_train, (X_train.shape[0], n_steps, 2))
    X_test = np.reshape(X_test, (X_test.shape[0], n_steps, 2))
    
    return X_train, y_train, X_test, y_test

def evaluate_config(config):
    
    n_steps, n_units, n_epochs, n_batch, n_layers = config
    
    X_train, y_train, X_test, y_test = get_dataset(n_steps)
    
    model = Sequential()
    model.add(Masking(mask_value=-100.0, input_shape=(None, X_train.shape[2])))
    if(n_layers == 1):
        model.add(LSTM(n_units, activation='relu', input_shape=(None, X_train.shape[2])))
    else:
        model.add(LSTM(n_units, activation='relu', input_shape=(None, X_train.shape[2]), return_sequences=True))
        
        while(n_layers-2):
            model.add(LSTM(n_units, activation='relu', return_sequences=True))
            n_layers -= 1
        model.add(LSTM(n_units, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=n_epochs, batch_size=n_batch)
    
    scores = model.evaluate(X_test, y_test, verbose=0)
    scores1 = model.evaluate(X_train, y_train, verbose=0)
#     print("Accuracy: %.2f%%" % (scores[1]*100))
    return np.array([round(scores[1]*100, 2), round(scores1[1]*100, 2)])


def get_configs():
    n_steps = [5,10,100,216]
    n_units = [10,100]
    n_epochs = [20,50]
    n_batch = [20,50]
    n_layers = [1, 2, 3]
    
    # create configs
    configs = list()
    for i in n_steps:
        for j in n_units:
            for k in n_epochs:
                for l in n_batch:
                    for m in n_layers:
                        cfg = [i, j, k, l, m]
                        configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs


def repeat_evaluate_config(config):
    n_avg = 2
    ans = np.array([0.0,0.0])
    
    for i in range(n_avg):
        ans += evaluate_config(config)
    ans /= n_avg
    
    return (config, ans)



configs = get_configs
# repeat_evaluate_config([5, 5, 5, 20, 1])
outfile = open('results/grid-search.txt', 'w')
# outfile.write(str(repeat_evaluate_config([5, 5, 2, 20, 1])))
# outfile.close()
for config in configs:
    out = repeat_evaluate_config(config)
    print(out)
    outfile.write(str(out))
outfile.close()
    










