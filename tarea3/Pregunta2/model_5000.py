import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from matplotlib import pyplot
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

from keras.datasets import imdb
np.random.seed(3)
srng = RandomStreams(8)
(X_train, y_train), (X_test, y_test) = imdb.load_data(seed=15)

# Concatenamiento de conjuntos de entrenamiento
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# Se cargan las 5000 palabras mas relevantes
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words, seed=15)

# Se acotan los comentarios a un maximo de 500 palabras
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)

# Tamanio vector generado por embedding
embedding_vector_length = 32

model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=500))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=64)
model.save('LSTM-words-5000.h5')