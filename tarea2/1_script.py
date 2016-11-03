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
    return (x_train, Y_train, y_train), (x_test, Y_test, y_test), (x_val, Y_val, y_val)


from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import SGD
import matplotlib.pyplot as plt


# Obtener conjuntos de datos

train, test, val = f()
x_train, Y_train, y_train = train
x_test, Y_test, y_test = test
x_val, Y_val, y_val = val

# Iterar sobre nuevas dimensionalidades
d_prime = [2, 8, 32, 64]
"""
for d in d_prime:
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
    autoencoder.save('1_1/basic_autoencoder_sigmoid_768x'+str(d)+'.h5')
    encoder.save('1_1/basic_encoder_sigmoid_768x'+str(d)+'.h5')
    decoder.save('1_1/basic_decoder_sigmoid_768x'+str(d)+'.h5')

# Pruebas con encoder relu y decoder sigmoid
for d in d_prime:
    input_img = Input(shape=(784,))
    encoded = Dense(d, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)
    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)
    encoded_input = Input(shape=(d,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    autoencoder.compile(optimizer=SGD(lr=1.0), loss='binary_crossentropy')
    autoencoder.fit(x_train,x_train,nb_epoch=50,batch_size=25,shuffle=True,
    validation_data=(x_val, x_val))
    autoencoder.save('1_1/basic_autoencoder_relusig_768x'+str(d)+'.h5')
    encoder.save('1_1/basic_encoder_relusig_768x'+str(d)+'.h5')
    decoder.save('1_1/basic_decoder_relusig_768x'+str(d)+'.h5')


# Pruebas con encoder sigmoid y decoder relu
for d in d_prime:
    input_img = Input(shape=(784,))
    encoded = Dense(d, activation='sigmoid')(input_img)
    decoded = Dense(784, activation='relu')(encoded)
    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)
    encoded_input = Input(shape=(d,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    autoencoder.compile(optimizer=SGD(lr=1.0), loss='binary_crossentropy')
    autoencoder.fit(x_train,x_train,nb_epoch=50,batch_size=25,shuffle=True,
    validation_data=(x_val, x_val))
    autoencoder.save('1_1/basic_autoencoder_sigrelu_768x'+str(d)+'.h5')
    encoder.save('1_1/basic_encoder_sigrelu_768x'+str(d)+'.h5')
    decoder.save('1_1/basic_decoder_sigrelu_768x'+str(d)+'.h5')
"""

# c) Calidad de representacion via kNN
from sklearn.neighbors import KNeighborsClassifier
from keras.models import load_model
from timeit import default_timer as timer

"""
print "Resultado en data original"
clf = KNeighborsClassifier(10)
print "Fitting..."
clf.fit(x_train, y_train)
start = timer()
print "Score..."
score = clf.score(x_test, y_test)
end = timer()
print 'Classification Accuracy %.2f' % score
print "Time: ", (end - start)


print "Resultados con AE Sigmoid"
for d in d_prime:
    autoencoder = load_model('1_1/basic_autoencoder_sigmoid_768x'+str(d)+'.h5')
    encoder = load_model('1_1/basic_encoder_sigmoid_768x'+str(d)+'.h5')
    encoded_train = encoder.predict(x_train)
    encoded_test = encoder.predict(x_test)
    clf = KNeighborsClassifier(10)
    print "Fitting..."
    clf.fit(encoded_train, y_train)
    start = timer()
    print "Score..."
    score = clf.score(encoded_test, y_test)
    end = timer()
    print 'Classification Accuracy %.2f' % score
    print "Time: ", (end - start)


print "Resultados con AE ReLu-Sigmoid"
for d in d_prime:
    autoencoder = load_model('1_1/basic_autoencoder_relusig_768x'+str(d)+'.h5')
    encoder = load_model('1_1/basic_encoder_relusig_768x'+str(d)+'.h5')
    encoded_train = encoder.predict(x_train)
    encoded_test = encoder.predict(x_test)
    clf = KNeighborsClassifier(10)
    print "Fitting..."
    clf.fit(encoded_train, y_train)
    start = timer()
    print "Score..."
    score = clf.score(encoded_test, y_test)
    end = timer()
    print 'Classification Accuracy %.2f' % score
    print "Time: ", (end - start)
"""

# d) Analisis calidad de agrupamiento
def clustering_accuracy(pred_labels,y,nclusters=10):
    true_pred = 0.0
    for i in range(0,nclusters):
        mvlabel = np.argmax(np.bincount(y[pred_labels==i]))
        true_pred += sum(y[pred_labels==i] == mvlabel)
    return true_pred/len(y)


from sklearn.cluster import KMeans
from sklearn import metrics
"""

print "kmeans relusig"
for d in d_prime:
    print "dim:",str(d)
    autoencoder = load_model('1_1/basic_autoencoder_relusig_768x'+str(d)+'.h5')
    encoder = load_model('1_1/basic_encoder_relusig_768x'+str(d)+'.h5')
    encoded_train = encoder.predict(x_train)
    encoded_test = encoder.predict(x_test)

    model = KMeans(n_clusters=10)
    labels_pred = model.fit_predict(encoded_train)
    score = metrics.adjusted_rand_score(y_train, labels_pred)
    print 'Clustering ARI %.2f' % score
    print 'Clustering ACC %.2f' % clustering_accuracy(labels_pred,y_train)

print "kmeans sigmoid"
for d in d_prime:
    print "dim:",str(d)
    autoencoder = load_model('1_1/basic_autoencoder_sigmoid_768x'+str(d)+'.h5')
    encoder = load_model('1_1/basic_encoder_sigmoid_768x'+str(d)+'.h5')
    encoded_train = encoder.predict(x_train)
    encoded_test = encoder.predict(x_test)

    model = KMeans(n_clusters=10)
    labels_pred = model.fit_predict(encoded_train)
    score = metrics.adjusted_rand_score(y_train, labels_pred)
    print 'Clustering ARI %.2f' % score
    print 'Clustering ACC %.2f' % clustering_accuracy(labels_pred,y_train)
"""



"""print "Raw data"
model = KMeans(n_clusters=10)
labels_pred = model.fit_predict(x_train)
score = metrics.adjusted_rand_score(y_train, labels_pred)
print 'Clustering ARI %.2f' % score
print 'Clustering ACC %.2f' % clustering_accuracy(labels_pred,y_train)"""

## e) Analisis via PCA
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Calidad de representacion
"""
for d in d_prime:
    print "Calidad representacion PCA d =",d
    pca = PCA(n_components=d)
    pca.fit(x_train)
    pca_train = pca.transform(x_train)
    pca_test = pca.transform(x_test)
    print "Reconstruction error %.2f" % pca.score(x_test)
    clf = KNeighborsClassifier(10)
    clf.fit(pca_train, y_train)
    score = clf.score(pca_test,y_test)
    print 'PCA SCORE %.2f' % score

    model = KMeans(n_clusters=10)
    labels_pred = model.fit_predict(pca_train)
    score = metrics.adjusted_rand_score(y_train, labels_pred)
    print 'Clustering ARI %.2f' % score
    print 'Clustering ACC %.2f' % clustering_accuracy(labels_pred, y_train)
"""

## f) Analisis via RBM
from sklearn.neural_network import BernoulliRBM
import numpy as np
import pickle

"""for d in d_prime:
    model = BernoulliRBM(n_components=d, batch_size=25,
    learning_rate=0.05,verbose=1, n_iter=50) ##n_components is d'
    model.fit(x_train)##Train using persistent Gibbs chains
    fileo = open('1_1/basicRBM_'+str(d)+'.pickle','wb')
    pickle.dump(model,fileo)
    fileo.close()"""

"""for d in d_prime:
    print "Calidad representacion RBM d =",d
    rbm = pickle.load(open("1_1/basicRBM_"+str(d)+".pickle", "rb"))
    rbm_train = rbm.transform(x_train)
    rbm_test = rbm.transform(x_test)
    print "Compressed dim:", float(rbm_test.shape[1])
    print "Original dim:", float(x_train.shape[1])
    print "Compresion:", float(x_train.shape[1])/float(rbm_test.shape[1])
    #print "Reconstruction error %.2f" % rbm.score(x_test)
    clf = KNeighborsClassifier(10)
    clf.fit(rbm_train, y_train)
    score = clf.score(rbm_test,y_test)
    print 'RBM SCORE %.2f' % score

    model = KMeans(n_clusters=10)
    labels_pred = model.fit_predict(rbm_train)
    score = metrics.adjusted_rand_score(y_train, labels_pred)
    print 'Clustering ARI %.2f' % score
    print 'Clustering ACC %.2f' % clustering_accuracy(labels_pred, y_train)"""

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from keras.models import load_model


## Modificacion de AE en Deep AE
new_d_prime = [2,4,8,16,32]
"""
## Caso L = 2
for d in new_d_prime:
    target_dim = d #try other and do a nice plot
    input_img = Input(shape=(784,))
    encoded1 = Dense(1000, activation='relu')(input_img)
    encoded2 = Dense(target_dim, activation='relu')(encoded1)
    decoded2 = Dense(1000, activation='relu')(encoded2)
    decoded1 = Dense(784, activation='sigmoid')(decoded2)
    autoencoder = Model(input=input_img, output=decoded1)
    encoder = Model(input=input_img, output=encoded1)
    autoencoder.compile(optimizer=SGD(lr=1.0), loss='binary_crossentropy')
    autoencoder.fit(x_train,x_train,nb_epoch=25,batch_size=25,shuffle=True, validation_data=(x_val, x_val))
    autoencoder.save('my_autoencoder_784x1000x'+str(d)+'.h5')
    encoder.save('my_encoder_784x1000x'+str(d)+'.h5')
    pca = PCA(n_components=target_dim)
    pca.fit(x_train)
    print "Reconstruction error en d=",d,":",pca.score(x_test)


## Caso L = 3
for d in new_d_prime:
    target_dim = d #try other and do a nice plot
    input_img = Input(shape=(784,))
    encoded1 = Dense(1000, activation='relu')(input_img)
    encoded2 = Dense(500, activation='relu')(encoded1)
    encoded3 = Dense(target_dim, activation='relu')(encoded2)
    decoded3 = Dense(500, activation='relu')(encoded3)
    decoded2 = Dense(1000, activation='relu')(decoded3)
    decoded1 = Dense(784, activation='sigmoid')(decoded2)
    autoencoder = Model(input=input_img, output=decoded1)
    encoder = Model(input=input_img, output=encoded2)
    autoencoder.compile(optimizer=SGD(lr=1.0), loss='binary_crossentropy')
    autoencoder.fit(x_train,x_train,nb_epoch=25,batch_size=25,shuffle=True, validation_data=(x_val, x_val))
    autoencoder.save('my_autoencoder_784x1000x500x'+str(d)+'.h5')
    encoder.save('my_encoder_784x1000x500x'+str(d)+'.h5')
    pca = PCA(n_components=target_dim)
    pca.fit(x_train)
    print "Reconstruction error en d=",d,":",pca.score(x_test)


## Caso L = 4
for d in new_d_prime:
    target_dim = d #try other and do a nice plot
    input_img = Input(shape=(784,))
    encoded1 = Dense(1000, activation='relu')(input_img)
    encoded2 = Dense(500, activation='relu')(encoded1)
    encoded3 = Dense(250, activation='relu')(encoded2)
    encoded4 = Dense(target_dim, activation='relu')(encoded3)
    decoded4 = Dense(250, activation='relu')(encoded4)
    decoded3 = Dense(500, activation='relu')(encoded3)
    decoded2 = Dense(1000, activation='relu')(decoded3)
    decoded1 = Dense(784, activation='sigmoid')(decoded2)
    autoencoder = Model(input=input_img, output=decoded1)
    encoder = Model(input=input_img, output=encoded3)
    autoencoder.compile(optimizer=SGD(lr=1.0), loss='binary_crossentropy')
    autoencoder.fit(x_train,x_train,nb_epoch=25,batch_size=25,shuffle=True, validation_data=(x_val, x_val))
    autoencoder.save('my_autoencoder_784x1000x500x250x'+str(d)+'.h5')
    encoder.save('my_encoder_784x1000x500x250x'+str(d)+'.h5')
    pca = PCA(n_components=target_dim)
    pca.fit(x_train)
    print "Reconstruction error en d=",d,":",pca.score(x_test)
"""

new_d_prime = [8,16,32]
for d in new_d_prime:
    print "Calidad representacion dAE d =",d,"y L=2"
    encoder = load_model('my_autoencoder_784x1000x'+str(d)+'.h5')
    #encoder = load_model('my_encoder_784x1000x'+str(d)+'.h5')
    print "Predicting"
    encoded_train = encoder.predict(x_train)
    encoded_test = encoder.predict(x_test)
    clf = KNeighborsClassifier(10)
    clf.fit(encoded_train, y_train)
    score = clf.score(encoded_test, y_test)
    print 'dAE kNN SCORE %.2f' % score
    model = KMeans(n_clusters=10)
    labels_pred = model.fit_predict(encoded_train)
    score = metrics.adjusted_rand_score(y_train, labels_pred)
    print 'Clustering ARI %.2f' % score
    print 'Clustering ACC %.2f' % clustering_accuracy(labels_pred,y_train)

    print "Calidad representacion dAE d =",d,"y L=3"
    dAE = load_model('my_autoencoder_784x1000x500x'+str(d)+'.h5')
    #encoder = load_model('my_encoder_784x1000x500x'+str(d)+'.h5')
    print "Predicting"
    encoded_train = encoder.predict(x_train)
    encoded_test = encoder.predict(x_test)
    clf = KNeighborsClassifier(10)
    clf.fit(encoded_train, y_train)
    score = clf.score(encoded_test, y_test)
    print 'dAE kNN SCORE %.2f' % score
    model = KMeans(n_clusters=10)
    labels_pred = model.fit_predict(encoded_train)
    score = metrics.adjusted_rand_score(y_train, labels_pred)
    print 'Clustering ARI %.2f' % score
    print 'Clustering ACC %.2f' % clustering_accuracy(labels_pred,y_train)


    print "Calidad representacion dAE d =",d,"y L=4"
    dAE = load_model('my_autoencoder_784x1000x500x250x'+str(d)+'.h5')
    #encoder = load_model('my_encoder_784x1000x500x250x'+str(d)+'.h5')
    print "Predicting"
    encoded_train = encoder.predict(x_train)
    encoded_test = encoder.predict(x_test)
    clf = KNeighborsClassifier(10)
    clf.fit(encoded_train, y_train)
    score = clf.score(encoded_test, y_test)
    print 'dAE kNN SCORE %.2f' % score
    model = KMeans(n_clusters=10)
    labels_pred = model.fit_predict(encoded_train)
    score = metrics.adjusted_rand_score(y_train, labels_pred)
    print 'Clustering ARI %.2f' % score
    print 'Clustering ACC %.2f' % clustering_accuracy(labels_pred,y_train)



"""
Calidad representacion dAE d = 2 y L=2
Predicting
dAE kNN SCORE 0.65
Clustering ARI 0.34
Clustering ACC 0.56
Calidad representacion dAE d = 2 y L=3
Predicting
dAE kNN SCORE 0.65
Clustering ARI 0.32
Clustering ACC 0.56
Calidad representacion dAE d = 2 y L=4
Predicting
dAE kNN SCORE 0.65
Clustering ARI 0.34
Clustering ACC 0.56
Calidad representacion dAE d = 4 y L=2
Predicting
dAE kNN SCORE 0.87
Clustering ARI 0.39
Clustering ACC 0.61
Calidad representacion dAE d = 4 y L=3
Predicting
dAE kNN SCORE 0.87

"""
