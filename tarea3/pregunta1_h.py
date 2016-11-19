from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

url = 'http://www.inf.utfsm.cl/~cvalle/international-airline-passengers.csv'
dataframe = pd.read_csv(url, sep=',', usecols=[1], engine='python', skipfooter=3)
dataframe[:] = dataframe[:].astype('float32')

df_train, df_test = dataframe[0:96].values, dataframe[96:].values

scaler = MinMaxScaler(feature_range=(0, 1)).fit(df_train)
stream_train_scaled = scaler.transform(df_train)
stream_test_scaled = scaler.transform(df_test)


def create_dataset(dataset, lag=1):
    dataX = np.zeros((dataset.shape[0]-lag, lag), dtype=np.float32)
    for i in range(lag):
        dataX[:,i] = dataset[i:-lag+i][:,0]
    dataY = dataset[lag:]
    return dataX, dataY

lag = 3
trainX, TrainY = create_dataset(stream_train_scaled, lag)
testX, TestY = create_dataset(stream_test_scaled, lag)

TrainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
TestX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


nb = range(4,13,2)
k = 5
kf_CV = KFold(TrainY[:,0].shape[0], k, shuffle=True)

results = []
for n in nb:
    print "Usando",n,"bloques LSTM"
    losses = []
    for i, (train, test) in enumerate(kf_CV):
        print "Analizando fold", i+1, "/", k
        model = None
        model = Sequential()
        model.add(LSTM(output_dim=n, input_dim=lag, activation='tanh', inner_activation='sigmoid'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(TrainX[train], TrainY[train], nb_epoch=100, batch_size=1, verbose=0)
        loss = model.evaluate(TrainX[test], TrainY[test])
        losses.append(loss)
    results.append(losses)
    print losses
print "Resultados finales"
print results
