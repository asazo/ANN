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


def process_set(PATH, i):
    f = os.path.join(PATH, 'data_batch_%d' % (i, ))
    datadict = unpickle(f)
    X = datadict['data'].T
    Y = np.array(datadict['labels'])
    Z = np.zeros((X.shape[0], X.shape[1] + 1))
    Z[:,:-1] = X
    Z[:, -1] = Y
    np.random.shuffle(Z)
    xtr.append(Z[5832:,0:-1])
    ytr.append(Z[5832:,-1])
    xval.append(Z[:5832,0:-1])
    yval.append(Z[:5832,-1])
    Xtr = np.concatenate(xtr)
    Ytr = np.concatenate(ytr)
    Xval = np.concatenate(xval)
    Yval = np.concatenate(yval)

def load_NORB_train_val(PATH, datarange=range(1, 11)):
    print "Loading training set..."
    xtr = []
    ytr = []
    xval = []
    yval = []
    for b in datarange:
        f = os.path.join(PATH, 'data_batch_%d' % (b, ))
        datadict = unpickle(f)
        X = datadict['data'].T
        Y = np.array(datadict['labels'])
        Z = np.zeros((X.shape[0], X.shape[1] + 1))
        Z[:,:-1] = X
        Z[:, -1] = Y
        np.random.shuffle(Z)
        xtr.append(Z[5832:,0:-1])
        ytr.append(Z[5832:,-1])
        xval.append(Z[:5832,0:-1])
        yval.append(Z[:5832,-1])
    Xtr = np.concatenate(xtr)
    Ytr = np.concatenate(ytr)
    Xval = np.concatenate(xval)
    Yval = np.concatenate(yval)

    del xtr,ytr,xval,yval
    print "Loaded."
    return Xtr, Ytr, Xval, Yval


def load_NORB_test(PATH):
    print "Loading testing set..."
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
    print "Loaded."
    return Xts, Yts


def scale_data(X, range=None):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    if not range:
        return StandardScaler().fit_transform(X)
    else:
        return MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)


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


# Establecer rangos para dividir training set en escenario no supervisado
def split_train(X, Y, theta):
    # n_s es la cantidad de ejemplos que si sabemos su etiqueta
    n_s = int(theta * n_tr)
    # Dividir training set
    X_s = X[0: n_s]
    Y_s = Y[0: n_s]
    X_ns = X[n_s: ]
    return X_s, Y_s, X_ns


# Cargar datos
Xtr, Ytr, Xval, Yval = load_NORB_train_val(".", range(1,4))
Xts, Yts = load_NORB_test(".")
n_tr = Xtr.shape[0]

# Escalar datos y categorizar
n_classes = 6
print "Scaling data..."
Xtr_scaled = scale_data(Xtr)
Xval_scaled = scale_data(Xval)
Xts_scaled = scale_data(Xts)
print "data scaled."
Ytr_class = np_utils.to_categorical(Ytr.astype(int), n_classes)
Yval_class = np_utils.to_categorical(Yval.astype(int), n_classes)
Yts_class = np_utils.to_categorical(Yts.astype(int), n_classes)


# Experimento: error de pruebas en funcion de theta (proporcion data no supervisada)
accuracies = []
for theta in np.linspace(0.1, 1, 10):
    Xtr_s, Ytr_s, Xtr_ns = split_train(Xtr_scaled, Ytr_class, theta)
    model = get_ff_model('relu', 6)
    model.fit(Xtr_s, Ytr_s, batch_size=10)
    a = model.evaluate(Xts_scaled, Yts_class, batch_size=10, verbose=1)
    print a
    accuracies.append(a)
print accuracies
