import os, random
import pickle as lpkl
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import confusion_matrix
import pickle as cPickle
import seaborn as sns
import tensorflow as tf
import tensorflow.keras.models as models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import utils
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Reshape, Flatten,Activation,Dropout, Conv2D, MaxPooling2D, ZeroPadding2D, LSTM, BatchNormalization, GaussianNoise
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import *
import keras
# Load the dataset ...
# You will need to separately download or generate this file from
# https://www.deepsig.io/datasets
# The file to get is RML2016.10a.tar.bz2
 
# you need an absolute path the file decompressed file so change the path.
# It is a pickle file for Python2 so a little extra code is needed to open it.
with open("/home/pov/.venv/intelintel/Qualification_Work/RDML/RML2016.10a_dict.pkl", 'rb') as f:
    Xd = cPickle.load(f, encoding="latin1") 
SNR = -20
for j in range(20):             
    del Xd[('AM-DSB', SNR)]
    del Xd[('AM-SSB', SNR)]
    del Xd[('WBFM', SNR)]                
    SNR = SNR + 2

snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
    # mod is the label. mod = modulation scheme
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        #snr = signal to noise ratio
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

# Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
# looks like taking half the samples for training
n_train = int(n_examples * 0.5)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
 
    data = list(yy)
 
    yy1 = np.zeros([len(data), max(data)+1])
    yy1[np.arange(len(data)),data] = 1
    return yy1
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

in_shp = list(X_train.shape[1:])
print (X_train.shape, in_shp)
classes = mods

dr = 0.1 # dropout rate (%)
model = Sequential()
model.add(Reshape(in_shp + [1], input_shape=(2, 128, 1)))
model.add(Conv2D(256, (1,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(1, 2), padding='valid',  data_format=None))
model.add(layers.Dropout(dr))

model.add(Conv2D(256, (1,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(1, 2), padding='valid',  data_format=None))
model.add(layers.Dropout(dr))

model.add(Conv2D(256, (1,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(1, 2), padding='valid',  data_format=None))
model.add(layers.Dropout(dr))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

nb_epoch = 100     # number of epochs to train on
batch_size = 200  # training batch size

# perform training ...
#   - call the main training loop in keras for our network+dataset
#weight written to jupyter directory (where notebook is). saved in hdf5 format.
filepath = '/home/pov/.venv/intelintel/Qualification_Work/RDML/convmodrecnets_CNN2_0.5.wts.h5'
#netron can open the h5 and show architecture of the neural network

history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=1,
    shuffle = True,
    validation_data=(X_test, Y_test),
    callbacks = [
        #params determine when to save weights to file. Happens periodically during fit.
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')
    ])
# we re-load the best weights once training is finished. best means lowest loss values for test/validation
model.load_weights(filepath)

score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
print (score)

plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.show()

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Reds, labels=[]):
    #plt.cm.Reds - color shades to use, Reds, Blues, etc.
    # made the image bigger- 800x800
    my_dpi=96
    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    #key call- data, how to interpolate thefp vakues, color map
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #adds a color legend to right hand side. Shows values for different shadings of blue.
    plt.colorbar()
    # create tickmarks with count = number of labels
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
#pass in X_test value and it predicts test_Y_hat
test_Y_hat = model.predict(X_test, batch_size=batch_size, verbose = 1)
#fill matrices with zeros
conf = np.zeros([len(classes),len(classes)])
#normalize confusion matrix
confnorm = np.zeros([len(classes),len(classes)])
 
#this puts all the data into an 11 x 11 matrix for plotting.
for i in range(0,X_test.shape[0]):
    # j is first value in list
    j = list(Y_test[i,:]).index(1)
    #np.argmax gives the index of the max value in the array, assuming flattened into single vector
    k = int(np.argmax(test_Y_hat[i,:]))
    #why add 1 to each value??
    conf[j,k] = conf[j,k] + 1
 
#takes the data to plot and normalizes it
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
print (confnorm)
print (classes)
plot_confusion_matrix(confnorm, labels=classes)

# Plot confusion matrix
acc = {}
 
#this create a new confusion matrix for each SNR
for snr in snrs:
 
    # extract classes @ SNR
    #changed map to list as part of upgrade from python2
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    
 
    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
 
    #create 11x11 matrix full of zeroes
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
 
    #normalize 0 .. 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
 
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    acc[snr] = 1.0*cor/(cor+ncor)


# Save results to a pickle file for plotting later
print (acc)
# Plot accuracy curve
# map function produces generator in python3 which does not work with plt. Need a list.
# list(map(chr,[66,53,0,94]))
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
import pandas