# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils



def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def load_single_NORB_train_val(PATH, i, onlyx=False):
    print "Cargando batch training set",i,"..."
    f = os.path.join(PATH, 'data_batch_%d' % (i, ))
    datadict = unpickle(f)
    X = datadict['data'].T
    Y = np.array(datadict['labels'])
    Z = np.zeros((X.shape[0], X.shape[1] + 1))
    Z[:,:-1] = X
    Z[:, -1] = Y
    np.random.shuffle(Z)
    if onlyx:
        Xtr = Z[5832:,0:-1]
        return Xtr
    else:
        Xtr = Z[5832:,0:-1]
        Ytr = Z[5832:,-1]
        Xval = Z[:5832,0:-1]
        Yval = Z[:5832,-1]
        print "Cargado"
        return Xtr, Ytr, Xval, Yval

def load_NORB_test(PATH):
    print "Cargando testing set..."
    xts = []
    yts = []
    for b in range(11, 13):
        f = os.path.join(PATH, 'data_batch_%d' % (b, ))
        datadict = unpickle(f)
        X = datadict['data'].T
        Y = np.array(datadict['labels'])
        Z = np.zeros((X.shape[0], X.shape[1] + 1))
        Z[:,:-1] = X
        Z[:, -1] = Y
        np.random.shuffle(Z)
        xts.append(Z[0:,0:-1])
        yts.append(Z[:,-1])
    Xts = np.concatenate(xts)
    Yts = np.concatenate(yts)

    del xts,yts
    print "Cargado."
    return Xts, Yts


def scale_data(X, normalize=True, myrange=None):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    if normalize and not myrange:
        print "Normalizando data (mean 0, std 1)"
        return StandardScaler().fit_transform(X)
    elif isinstance(myrange, tuple):
        print "Escalando data al rango", myrange
        return X * (myrange[1] - myrange[0]) + myrange[0]
    else:
        return "Error mientras escalaba."


# Modelo MLP FF
def get_ff_model(activation, n_classes):
    model = Sequential()
    model.add(Dense(4000, input_dim=2048, activation=activation))
    model.add(Dense(2000, activation=activation))
    model.add(Dense(n_classes, activation='softmax'))
    sgd = SGD(lr=0.1, decay=0.0)
    model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model


# Definir numero de clases
n_classes = 6
Xts, Yts = load_NORB_test(".")
Xts_scaled = scale_data(Xts)
Yts_class = np_utils.to_categorical(Yts.astype(int), n_classes)


accuracies = []
activation = 'relu'
hidden_layer = 512
hidden_layer2 = 100

model = Sequential()
model.add(Dense(hidden_layer, input_dim=2048, activation=activation))
model.add(Dense(hidden_layer2, activation=activation))
model.add(Dense(n_classes, activation='softmax'))
sgd = SGD(lr=0.1, decay=0.0)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

for i, theta in enumerate(np.linspace(0.1, 1, 10)):
    print "Entrenando..."
    for n in range(10):
        for k in range(0, i+1):
            print "Leyendo batch",k
            Xtr, Ytr, Xval, Yval = load_single_NORB_train_val(".", k+1)
            # Escalar datos y categorizar
            print "Escalando data..."
            Xtr_scaled = scale_data(Xtr)
            Xval_scaled = scale_data(Xval)
            print "Data escalada."
            print "Pasando a data categorica para labels..."
            Ytr_class = np_utils.to_categorical(Ytr.astype(int), n_classes)
            Yval_class = np_utils.to_categorical(Yval.astype(int), n_classes)
            print "Data categorizada."
            model.fit(Xtr_scaled, Ytr_class, batch_size=10,
                      validation_data=(Xval_scaled, Yval_class), nb_epoch=1)
            print "Batch entrenado."

    a = model.evaluate(Xts_scaled, Yts_class, batch_size=10, verbose=1)
    print "Resultado:",a
    accuracies.append(a)
    print accuracies
    del Xtr, Ytr, Xval, Yval
print accuracies
