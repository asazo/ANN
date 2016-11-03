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

# Experimento: error de pruebas en funcion de theta (proporcion data no supervisada)
# Entrenar por porcentajes conocidos implica iterativamente avanzar batch sobre batch...

accuracies = []
model = get_ff_model('relu', n_classes)
print "Metricas:",model.metrics_names

for i, theta in enumerate(np.linspace(0.1, 1, 10)):
    print "Analizando theta =",theta
    print "Utilizando",i+1,"batches de 10"
    Xtr, Ytr, Xval, Yval = load_single_NORB_train_val(".", i+1)
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
    model.fit(Xtr_scaled, Ytr_class, batch_size=10, validation_data=(Xval_scaled, Yval_class), nb_epoch=1)
    print "Batch entrenado."
    a = model.evaluate(Xts_scaled, Yts_class, batch_size=10, verbose=1)
    print "Resultado:",a
    accuracies.append(a)
print accuracies


from sklearn.neural_network import BernoulliRBM
from sklearn.externals import joblib
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD

# Pre entrenar con RBM
RBM1 = BernoulliRBM(n_components=512, batch_size=10,
                            learning_rate=0.01, verbose=1, n_iter=30)
RBM2 = BernoulliRBM(n_components=100, batch_size=10,
                            learning_rate=0.01, verbose=1, n_iter=30)

for i, rev_theta in enumerate(np.linspace(0.1, 1, 10)):
    theta = 1 - rev_theta
    print "Preentrenando modelo para theta=",theta
    print "Leyendo batch",i+1
    Xtr_ns = load_single_NORB_train_val(".", i+1, onlyx=True)
    Xtr_ns = scale_data(Xtr_ns)
    RBM1.partial_fit(Xtr_ns)
    Xtr_ns2 = RBM1.transform(Xtr_ns)
    print "..."
    Xtr_ns2 = scale_data(Xtr_ns2)
    RBM2.partial_fit(Xtr_ns2)
    del Xtr_ns, Xtr_ns2
    print "..."
    joblib.dump(RBM1, "2/RBM1_512_"+str(theta)+".pkl")
    joblib.dump(RBM2, "2/RBM2_512_"+str(theta)+".pkl")


# Entrenar usando RBM
accuracies = []
activation = 'tanh'

for i, theta in enumerate(np.linspace(0.1, 1, 10)):
    print "Analizando theta =",theta
    if i != 9:
        print "Cargando rbms"
        RBM1 = joblib.load('2/RBM1_512_'+str(theta)+".pkl")
        RBM2 = joblib.load('2/RBM2_512_'+str(theta)+".pkl")

    model = Sequential()
    model.add(Dense(512, input_dim=2048, activation=activation))
    if i != 9:
        print "seteando pesos 1"
        model.layers[-1].set_weights([RBM1.components_.T, RBM1.intercept_hidden_])
    model.add(Dense(100, activation=activation))
    if i != 9:
        print "seteando pesos 2"
        model.layers[-1].set_weights([RBM2.components_.T, RBM2.intercept_hidden_])
    model.add(Dense(n_classes, activation='softmax'))
    sgd = SGD(lr=0.1, decay=0.0)
    model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

    print "Entrenando..."
    for n in range(2):
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

# Preentrenamiento con AE


hidden_layer = 512
hidden_layer2 = 100

activation1 = 'relu'
activation2 = 'sigmoid'


input_img1 = Input(shape=(2048,))
encoded1 = Dense(hidden_layer, activation=activation1)(input_img1)
decoded1 = Dense(2048, activation=activation2)(encoded1)
autoencoder1 = Model(input=input_img1, output=decoded1)
encoder1 = Model(input=input_img1, output=encoded1)
autoencoder1.compile(optimizer=SGD(lr=0.001), loss='binary_crossentropy')

input_img2 = Input(shape=(hidden_layer,))
encoded2 = Dense(hidden_layer2, activation=activation2)(input_img2)
decoded2 = Dense(hidden_layer, activation=activation2)(encoded2)
autoencoder2 = Model(input=input_img2, output=decoded2)
encoder2 = Model(input=input_img2, output=encoded2)
autoencoder2.compile(optimizer=SGD(lr=0.001), loss='binary_crossentropy')


for i, rev_theta in enumerate(np.linspace(0.1, 0.9, 9)):
    theta = 1 - rev_theta
    print "Preentrenando modelo para theta=",theta
    print "Leyendo batch",i+1
    Xtr_ns, Ytr_ns, Xval_ns, Yval_ns = load_single_NORB_train_val(".", i+1)
    Xtr_ns = scale_data(Xtr_ns)
    Xval_ns = scale_data(Xval_ns)

    autoencoder1.fit(Xtr_ns, Xtr_ns, nb_epoch=10, batch_size=250,shuffle=True, validation_data=(Xval_ns, Xval_ns))
    autoencoder1.save('2/AE1_'+str(theta)+'.h5')
    encoder1.save('2/E1_'+str(theta)+'.h5')

    Xtr_ns_1 = encoder1.predict(Xtr_ns)
    Xval_ns_1 = encoder1.predict(Xval_ns)
    autoencoder2.fit(Xtr_ns_1,Xtr_ns_1,nb_epoch=10,batch_size=250, shuffle=True, validation_data=(Xval_ns_1, Xval_ns_1))
    autoencoder2.save('2/AE2_'+str(theta)+'.h5')
    encoder2.save('2/E2_'+str(theta)+'.h5')


accuracies = []
activation = 'tanh'
from keras.models import load_model


for i, theta in enumerate(np.linspace(0.1, 1, 10)):
    print "Analizando theta =",theta
    if i != 9:
        AE1 = load_model('2/AEpretraining512/AE1_'+str(theta)+".h5")
        AE2 = load_model('2/AEpretraining512/AE2_'+str(theta)+".h5")

    model = Sequential()
    model.add(Dense(hidden_layer, input_dim=2048, activation=activation))
    if i != 9:
        print "setear pesos 1"
        model.layers[-1].set_weights(AE1.layers[1].get_weights())
    model.add(Dense(hidden_layer2, activation=activation))
    if i != 9:
        print "setear pesos 2"
        model.layers[-1].set_weights(AE2.layers[1].get_weights())
    model.add(Dense(n_classes, activation='softmax'))
    sgd = SGD(lr=0.1, decay=0.0)
    model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

    print "Entrenando..."
    for n in range(2):
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
