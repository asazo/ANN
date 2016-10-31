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


def load_single_NORB_train_val(PATH, i):
    print "Cargando batch training set",i,"..."
    f = os.path.join(PATH, 'data_batch_%d' % (i, ))
    datadict = unpickle(f)
    X = datadict['data'].T
    Y = np.array(datadict['labels'])
    Z = np.zeros((X.shape[0], X.shape[1] + 1))
    Z[:,:-1] = X
    Z[:, -1] = Y
    np.random.shuffle(Z)
    Xtr = Z[5832:,0:-1]
    Ytr = Z[5832:,-1]
    Xval = Z[:5832,0:-1]
    Yval = Z[:5832,-1]
    print "Cargado"
    return Xtr, Ytr, Xval, Yval


"""def load_NORB_train_val(PATH, datarange=range(1, 11)):
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
    return Xtr, Ytr, Xval, Yval"""


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


# Establecer rangos para dividir training set en escenario no supervisado
def split_train(X, Y, theta):
    # n_s es la cantidad de ejemplos que si sabemos su etiqueta
    n_s = int(theta * n_tr)
    # Dividir training set
    X_s = X[0: n_s]
    Y_s = Y[0: n_s]
    X_ns = X[n_s: ]
    return X_s, Y_s, X_ns

# Definir numero de clases
n_classes = 6
# Cargar datos de prueba.
print "Cargando"
Xts, Yts = load_NORB_test(".")
Xts_scaled = scale_data(Xts)
Yts_class = np_utils.to_categorical(Yts.astype(int), n_classes)

# Experimento: error de pruebas en funcion de theta (proporcion data no supervisada)
accuracies = []
for theta in np.linspace(0.1, 1, 10):
    model = get_ff_model('relu', n_classes)
    print "Metricas:",model.metrics_names
    # Entrenamiento batch a batch
    for i in range(1, 11):
        print "Entrenando batch",i,"de 10"
        Xtr, Ytr, Xval, Yval = load_single_NORB_train_val(".", i)
        n_tr = Xtr.shape[0]
        # Escalar datos y categorizar
        print "Escalando data..."
        Xtr_scaled = scale_data(Xtr)
        Xval_scaled = scale_data(Xval)
        print "Data escalada."
        print "Pasando a data categorica para labels..."
        Ytr_class = np_utils.to_categorical(Ytr.astype(int), n_classes)
        Yval_class = np_utils.to_categorical(Yval.astype(int), n_classes)
        print "Data categorizada."
        Xtr_s, Ytr_s, Xtr_ns = split_train(Xtr_scaled, Ytr_class, theta)
        model.fit(Xtr_s, Ytr_s, batch_size=10, validation_data=(Xval_scaled, Yval_class))
        print "Batch entrenado."
    print "Evaluando modelo para theta =",theta
    a = model.evaluate(Xts_scaled, Yts_class, batch_size=10, verbose=1)
    print a
    accuracies.append(a)
print accuracies
