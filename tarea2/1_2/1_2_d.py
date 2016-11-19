from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from numpy.random import binomial
from numpy.random import standard_normal

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

# Obtener conjuntos de datos
train, test, val = f()
x_train, Y_train = train
x_test, Y_test = test
x_val, Y_val = val

devsts = [0.1,0.3,0.5,0.7,0.9,1.0]
i = 0

for devst in devsts:
    i += 1
    noise_mask = devst*standard_normal(size=x_train.shape)
    noisy_x_train = x_train+noise_mask
    noise_mask = devst*standard_normal(size=x_val.shape)
    noisy_x_val = x_val+noise_mask
    noise_mask = devst*standard_normal(size=x_test.shape)
    noisy_x_test = x_test+noise_mask

    input_img = Input(shape=(784,))
    encoded = Dense(32, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)
    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)
    encoded_input = Input(shape=(32,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    autoencoder.compile(optimizer=SGD(lr=1.0), loss='binary_crossentropy')
    autoencoder.fit(noisy_x_train,x_train,nb_epoch=50,batch_size=25,shuffle=True, validation_data=(noisy_x_val, x_val))
    #model_json = autoencoder.to_json()
    #with open("model_2_"+str(i)+".json", "w") as json_file:
    #    json_file.write(model_json)
    # serialize weights to HDF5
    #model.save_weights("model_2_"+str(i)+".h5")

    autoencoder.save("autoencoder_2_"+str(i)+".h5")
    encoder.save("encoder_2_"+str(i)+".h5")
    decoder.save("decoder_2_"+str(i)+".h5")
