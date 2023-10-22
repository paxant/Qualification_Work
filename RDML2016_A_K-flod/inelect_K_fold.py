import os, random
import pickle as lpkl
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import confusion_matrix
import pickle as cPickle
import seaborn as sns
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
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
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
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
def create_model():
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

    model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    return model

nb_epoch = 200     # number of epochs to train on
batch_size = 200  # training batch size

# perform training ...
#   - call the main training loop in keras for our network+dataset
#weight written to jupyter directory (where notebook is). saved in hdf5 format.
filepath = '/home/pov/.venv/intelintel/Qualification_Work/RDML/'
#netron can open the h5 and show architecture of the neural network

n_examples = X.shape[0]
# looks like taking half the samples for training

N_DATAS = int(n_examples * 1)
DATAS_idx = np.random.choice(range(0,n_examples), size=N_DATAS, replace=False)
X_DATAS = X[DATAS_idx]

def to_onehot(yy):
    data = list(yy)
    yy1 = np.zeros([len(data), max(data)+1])
    yy1[np.arange(len(data)),data] = 1
    return yy1

Y_DATAS = to_onehot(map(lambda x: mods.index(lbl[x][0]), DATAS_idx))

in_shp = list(X_DATAS.shape[1:])

print(in_shp)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
check_model = int(0)
for train, test_valid in kfold.split(X_DATAS, Y_DATAS, groups=None):
    model = create_model()
    
    file_Name_Train_Dataset = filepath + "Train_Dataset_CNN_10_Fold_cross_valid_" + str(check_model)
    file_Name_Test_Dataset = filepath + "Test_Dataset_CNN_10_Fold_cross_valid_" + str(check_model)
    file_Name_CNN = filepath + "CNN_Model_10_fold_cross_vaild_" + str(check_model) + ".wts.h5"
    file_Name_CNN_history = filepath + "CNN_Model_10_fold_cross_vaild_history_" + str(check_model)

    Dataset_train = {}
    Dataset_train[('X')] = X_DATAS[train]
    Dataset_train[('Y')] = Y_DATAS[train]

    outfile = open(file_Name_Train_Dataset,'wb')      # Вывод в файл датасета
    lpkl.dump(Dataset_train,outfile)                       # Байтовая запись в файл
    outfile.close() 

    Dataset_test = {}
    Dataset_test[('X')] = X_DATAS[test_valid]
    Dataset_test[('Y')] = Y_DATAS[test_valid]

    outfile = open(file_Name_Test_Dataset,'wb')      # Вывод в файл датасета
    lpkl.dump(Dataset_test, outfile)                       # Байтовая запись в файл
    outfile.close() 

    history = model.fit(X_DATAS[train],
            Y_DATAS[train],
            batch_size=batch_size,
            epochs=nb_epoch,
            verbose=1,
            shuffle = True,
            validation_data=(X_DATAS[test_valid], Y_DATAS[test_valid]),
            callbacks = [
                #params determine when to save weights to file. Happens periodically during fit.
                keras.callbacks.ModelCheckpoint(file_Name_CNN, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')
            ])
    
    score = model.evaluate(X_DATAS[test_valid], Y_DATAS[test_valid], verbose=1, batch_size=batch_size)
    print (score)

    plt.figure()
    plt.title('Training performance' + str(check_model))
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()
    plt.show()

    outfile = open(file_Name_CNN_history,'wb')      # Вывод в файл датасета
    lpkl.dump(history, outfile)                       # Байтовая запись в файл
    outfile.close()

    check_model = check_model + 1
    del history
    del model

import pandas