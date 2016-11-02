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

# c) Calidad de representacion via kNN
from sklearn.neighbors import KNeighborsClassifier
from keras.models import load_model
from timeit import default_timer as timer


"""print "Resultado en data original"
clf = KNeighborsClassifier(10)
print "Fitting..."
clf.fit(x_train, y_train)
start = timer()
print "Score..."
score = clf.score(x_test, y_test)
end = timer()
print 'Classification Accuracy %.2f' % score
print "Time: ", (end - start)"""
"""
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
    print 'Clustering ACC %.2f' % clustering_accuracy(labels_pred,y_train)"""

"""print "kmeans sigmoid"
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
    print 'Clustering ACC %.2f' % clustering_accuracy(labels_pred,y_train)"""



"""
print "Raw data"
model = KMeans(n_clusters=10)
labels_pred = model.fit_predict(x_train)
score = metrics.adjusted_rand_score(y_train, labels_pred)
print 'Clustering ARI %.2f' % score
print 'Clustering ACC %.2f' % clustering_accuracy(labels_pred,y_train)"""


"""## e) Analisis via PCA
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Calidad de representacion
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
    print 'Clustering ACC %.2f' % clustering_accuracy(labels_pred, y_train)"""


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


