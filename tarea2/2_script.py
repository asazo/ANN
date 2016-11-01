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

"""accuracies = []
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
print accuracies"""


from sklearn.neural_network import BernoulliRBM
from sklearn.externals import joblib

# Pre entrenar con RBM
"""RBM1 = BernoulliRBM(n_components=4000, batch_size=2916,
                            learning_rate=0.01, verbose=1, n_iter=30)
RBM2 = BernoulliRBM(n_components=2000, batch_size=2916,
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
    joblib.dump(RBM1, "2/RBM1_"+str(theta)+".pkl")
    joblib.dump(RBM2, "2/RBM2_"+str(theta)+".pkl")
"""

# Entrenar usando RBM
"""accuracies = []
activation = 'relu'

for i, theta in enumerate(np.linspace(0.1, 1, 10)):
    print "Analizando theta =",theta
    if i == 10:
        RBM1 = joblib.load('2/RBM1_'+str(theta)+".pkl")
        RBM2 = joblib.load('2/RBM2_'+str(theta)+".pkl")

    model = Sequential()
    model.add(Dense(4000, input_dim=2048, activation=activation))
    if i == 10:
        model.layers[-1].set_weights([RBM1.components_.T, RBM1.intercept_hidden_])
    model.add(Dense(2000, activation=activation))
    if i == 10:
        model.layers[-1].set_weights([RBM2.components_.T, RBM2.intercept_hidden_])
    model.add(Dense(n_classes, activation='softmax'))
    sgd = SGD(lr=0.1, decay=0.0)
    model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

    print "Entrenando..."
    for n in range(1):
        for k in range(1, 11):
            print "Leyendo batch",k
            Xtr, Ytr, Xval, Yval = load_single_NORB_train_val(".", k)
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
    del Xtr, Ytr, Xval, Yval
print accuracies
"""

# Preentrenamiento con AE
"""RBM1 = BernoulliRBM(n_components=4000, batch_size=2916,
                            learning_rate=0.01, verbose=1, n_iter=30)
RBM2 = BernoulliRBM(n_components=2000, batch_size=2916,
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
    joblib.dump(RBM1, "2/RBM1_"+str(theta)+".pkl")
    joblib.dump(RBM2, "2/RBM2_"+str(theta)+".pkl")
"""

activation = 'relu'
input_img1 = Input(shape=(2048,))
encoded1 = Dense(4000, activation=activation)(input_img1)
decoded1 = Dense(784, activation='sigmoid')(encoded1)
autoencoder1 = Model(input=input_img1, output=decoded1)
encoder1 = Model(input=input_img1, output=encoded1)
autoencoder1.compile(optimizer=SGD(lr=1.0), loss='binary_crossentropy')

for i, rev_theta in enumerate(np.linspace(0.1, 1, 10)):
    print "Preentrenando modelo para theta=",theta
    print "Leyendo batch",i+1
    Xtr_ns = load_single_NORB_train_val(".", i+1, onlyx=True)
    Xtr_ns = scale_data(Xtr_ns)
    theta = 1 - rev_theta
    autoencoder1.fit(Xtr_ns, Xtr_ns, nb_epoch=25, batch_size=25,
    shuffle=True, validation_data=(x_val, x_val))
    encoded_input1 = Input(shape=(4000,))
    autoencoder1.save('AE1_'+str(theta)+'.h5')
    encoder1.save('E1'+str(theta)+'.h5')

    Xtr_ns_2 = encoder1.predict(Xtr_ns)
    x_val_encoded1 = encoder1.predict(x_val)
    x_test_encoded1 = encoder1.predict(x_test)

    input_img2 = Input(shape=(n_hidden_layer1,))
    encoded2 = Dense(n_hidden_layer2, activation=activation_layer2)(input_img2)
    decoded2 = Dense(n_hidden_layer2, activation=decoder_activation_2)(encoded2)
    autoencoder2 = Model(input=input_img2, output=decoded2)
    encoder2 = Model(input=input_img2, output=encoded2)
    autoencoder2.compile(optimizer=optimizer_, loss=loss_)
    autoencoder2.fit(x_train_encoded1,x_train_encoded1,nb_epoch=epochs_,batch_size=batch_size_,
    shuffle=True, validation_data=(x_val_encoded1, x_val_encoded1))
    encoded_input2 = Input(shape=(n_hidden_layer2,))
    autoencoder2.save('autoencoder_layer2.h5')
    encoder2.save('encoder_layer2.h5')
