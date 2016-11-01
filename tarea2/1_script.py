from keras.datasets import mnist
import numpy as np

def f(nval=1000):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    x_val = x_train[-nval:]
    y_val = y_train[-nval:]
    x_train = x_train[:-nval]
    y_train = y_train[:-nval]
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_val = np_utils.to_categorical(y_val, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    return (x_train, Y_train), (x_test, Y_test), (x_val, Y_val)


from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import SGD
import matplotlib.pyplot as plt


# Obtener conjuntos de datos

train, test, val = f()
x_train, Y_train = train
x_test, Y_test = test
x_val, Y_val = val

# Iterar sobre nuevas dimensionalidades
d_prime = [2, 8, 32, 64]

"""for d in d_prime:
    input_img = Input(shape=(784,))
    encoded = Dense(d, activation='sigmoid')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)
    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)
    encoded_input = Input(shape=(d,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    autoencoder.compile(optimizer=SGD(lr=1.0), loss='binary_crossentropy')
    hist = autoencoder.fit(x_train,x_train,nb_epoch=50,batch_size=25,shuffle=True,
    validation_data=(x_val, x_val))
    autoencoder.save('1_1/basic_autoencoder_relu_768x'+str(d)+'.h5')
    autoencoder.save('1_1/basic_encoder_relu_768x'+str(d)+'.h5')
    autoencoder.save('1_1/basic_decoder_relu_768x'+str(d)+'.h5')"""

# Pruebas con encoder relu y decoder sigmoid
"""for d in d_prime:
    input_img = Input(shape=(784,))
    encoded = Dense(d, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)
    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)
    encoded_input = Input(shape=(d,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    autoencoder.compile(optimizer=SGD(lr=1.0), loss='binary_crossentropy')
    hist = autoencoder.fit(x_train,x_train,nb_epoch=50,batch_size=25,shuffle=True,
    validation_data=(x_val, x_val))
    autoencoder.save('1_1/basic_autoencoder_relusig_768x'+str(d)+'.h5')
    autoencoder.save('1_1/basic_encoder_relusig_768x'+str(d)+'.h5')
    autoencoder.save('1_1/basic_decoder_relusig_768x'+str(d)+'.h5')"""


# Pruebas con encoder sigmoid y decoder relu
"""for d in d_prime:
    input_img = Input(shape=(784,))
    encoded = Dense(d, activation='sigmoid')(input_img)
    decoded = Dense(784, activation='relu')(encoded)
    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)
    encoded_input = Input(shape=(d,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    autoencoder.compile(optimizer=SGD(lr=1.0), loss='binary_crossentropy')
    hist = autoencoder.fit(x_train,x_train,nb_epoch=50,batch_size=25,shuffle=True,
    validation_data=(x_val, x_val))
    autoencoder.save('1_1/basic_autoencoder_sigrelu_768x'+str(d)+'.h5')
    autoencoder.save('1_1/basic_encoder_sigrelu_768x'+str(d)+'.h5')
    autoencoder.save('1_1/basic_decoder_sigrelu_768x'+str(d)+'.h5')"""

from sklearn.neighbors import KNeighborsClassifier
from keras.models import load_model
from timeit import default_timer as timer


print "Resultado en data original"
clf = KNeighborsClassifier(10)
print "Fitting..."
clf.fit(x_train, Y_train)
start = timer()
print "Score..."
score = clf.score(x_test, Y_test)
end = timer()
print 'Classification Accuracy %.2f' % score
print "Time: ", (end - start)

print "Resultados con AE Sigmoid"
for d in dprime:
    autoencoder = load_model('1_1/basic_autoencoder_sigmoid_768x'+str(d)+'.h5')
    encoder = load_model('1_1/basic_encoder_sigmoid_768x'+str(d)+'.h5')
    encoded_train = encoder.predict(x_train)
    encoded_test = encoder.predict(x_test)
    clf = KNeighborsClassifier(10)
    print "Fitting..."
    clf.fit(encoded_train, Y_train)
    start = timer()
    print "Score..."
    score = clf.score(encoded_test, Y_test)
    end = timer()
    print 'Classification Accuracy %.2f' % score
    print "Time: ", (end - start)


print "Resultados con AE ReLu-Sigmoid"
for d in dprime:
    autoencoder = load_model('1_1/basic_autoencoder_relusig_768x'+str(d)+'.h5')
    encoder = load_model('1_1/basic_encoder_relusig_768x'+str(d)+'.h5')
    encoded_train = encoder.predict(x_train)
    encoded_test = encoder.predict(x_test)
    clf = KNeighborsClassifier(10)
    print "Fitting..."
    clf.fit(encoded_train, Y_train)
    start = timer()
    print "Score..."
    score = clf.score(encoded_test, Y_test)
    end = timer()
    print 'Classification Accuracy %.2f' % score
    print "Time: ", (end - start)
