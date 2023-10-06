# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/home/pov/.venv/intelintel/Qualification_Work/RDML'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import tensorflow as tf
from tensorflow.keras.layers import Reshape, Conv2D, MaxPool2D, ZeroPadding2D, Dense, Dropout, Activation, Flatten, GaussianNoise

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pickle


def digitizer(labels):
    
    unique_labels = np.unique(labels)
    label_dict = {}
    num = 1
    for i in unique_labels:
        label_dict[i] = num
        num += 1 
    
    digit_label = []
    for i in labels:
        digit_label.append(label_dict[i])
    
    return label_dict,  digit_label 
    
            
def onehot_encoder(L_dict,  labels):
    num_classes = len(L_dict)
    vector = np.zeros(num_classes)
    vector[L_dict[labels]-1] = 1
    return vector



def confusion_matrix_create (y_true, y_pred, labels_dict, title):
    
    labels = []
    for i in labels_dict.items():
        labels.append(i[0])
    y_true = np.argmax(y_true, axis =1)
    y_true = np.array(y_true) + 1
    y_pred = np.array(y_pred) + 1
    
    
    
    updated_pred = []
    updated_true = []

    for i in range(len(y_true)):

        for key,value in labels_dict.items():
            if value == y_true[i]:
                updated_true.append(key)

            if value == y_pred[i]:
                updated_pred.append(key)

    cm = confusion_matrix(updated_true,updated_pred, labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.xticks(ticks=[-1,0,1,2,3,4,5,6,7,8,9,10], rotation=45)
    plt.yticks(ticks=[-1,0,1,2,3,4,5,6,7,8,9,10], rotation=45)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    ax.set (title=title, 
            ylabel='True label',
            xlabel='Predicted label')
    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.show()


    import numpy as np
import pickle 

with open('/home/pov/.venv/intelintel/Qualification_Work/RDML/RML2016.10a_dict.pkl', "rb") as p:
    d = pickle.load(p, encoding='latin1')

classes = []    
for i in d.keys():    
    if i[0] not in classes:
        classes.append(i[0])

# creating class dictionary for strings to digits transformation.
label_dict,  digit_label = digitizer(classes)



SNRs = {}
for key in d.keys():
    if key not in SNRs:
        SNRs[key[1]] = []
SNRs.keys()



j = 0
for keys in d.keys():
    for arrays in d[keys]:
        # convert labels to one-hot encoders.
        SNRs[keys[1]].append([onehot_encoder(label_dict, keys[0]),np.array(arrays)]) 

outfile = open('dataset','wb')
pickle.dump(SNRs,outfile)
outfile.close()


outfile = open('class_dict','wb')
pickle.dump(label_dict,outfile)
outfile.close()

import pickle
import itertools
from random import shuffle

with open('dataset', 'rb') as file:
    data = pickle.load(file, encoding='Latin')

for key in data.keys():
    shuffle(data[key])

new_data = {'combined': []}
SNR_test = {}

for key in data.keys():
    train_len = int(0.9 * len(data[key]))
    new_data['combined'].append(data[key][:train_len])
    SNR_test[key] = data[key][train_len:]

new_data['combined'] = list(itertools.chain.from_iterable(new_data['combined']))

outfile = open('new_model_SNR_test_samples', 'wb')
pickle.dump(SNR_test, outfile)
outfile.close()

outfile = open('combined_SNR_data', 'wb')
pickle.dump(new_data, outfile)
outfile.close()


from keras.datasets import cifar10
from tensorflow.keras import utils # строка изменена
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LSTM, BatchNormalization
from keras import metrics
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
import pickle
import matplotlib.pyplot as plt
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.callbacks import EarlyStopping

def CLDNN():
    model = Sequential()
    model.add(Conv2D(60, 10, activation='relu', padding='same',input_shape=(2, 128, 1)))
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid',  data_format=None))
    model.add(layers.Dropout(0.3))

    model.add(Conv2D(60, 10, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid',  data_format=None))
    model.add(layers.Dropout(0.3))
    
    model.add(Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(Reshape((2, 4096)))

    model.add(LSTM(128,activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(Dense(11, activation='softmax'))

    return model

def Robust_CNN():
    
    model = Sequential()
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(2,128,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid',  data_format=None))
    model.add(layers.Dropout(.3))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid', data_format=None))
    model.add(layers.Dropout(.3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid', data_format=None))
    model.add(layers.Dropout(.3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 2), padding='valid', data_format=None))
    model.add(layers.Dropout(.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(11, activation='softmax'))
    
    return model

def DNN():
    model = Sequential()
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(250,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(11,activation='softmax'))
    return model

from sklearn.metrics import confusion_matrix
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
import os
import pickle
import numpy as np
# Use this code only if you want to generate 20 different models corresponding to 20 SNR values
accuracies_All = []
confusion_matrices_All = []

for key in SNRs.keys():
    dataset = []
    labels = []

    for values in SNRs[key]:
        labels.append(values[0])
        dataset.append(values[1])

    print('Starting training for SNR:', key)

    N = len(dataset)
    shuffled_indeces = np.random.permutation(range(N))
    new_dataset = np.array(dataset)[shuffled_indeces,:,:]
    new_labels = np.array(labels)[shuffled_indeces,:]

    num_train = int(0.8*N)
    x_train = new_dataset[:num_train,:,:]
    y_train = new_labels[:num_train,:]

    num_val = int(0.1*len(x_train))

    x_val = x_train[:num_val,:,:]
    x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2], -1)
    y_val = y_train[:num_val,:]

    x_train = x_train[num_val:,:,:]
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2], -1)
    y_train = y_train[num_val:,:]

    x_test = new_dataset[num_train:,:,:]
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2], -1)
    y_test = new_labels[num_train:,:]

    models = CLDNN()
    opt = Adam(learning_rate=0.0003)
    models.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    num_epochs = 300

    # Checkpoint for models
    ckpt_folder = "cldnn_models/"
    ckpt_file_path = 'cldnn_model_SNR_{}'.format(key)
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    model_ckpt_callback = ModelCheckpoint(filepath=ckpt_folder+ckpt_file_path,monitor='val_loss', mode='min', save_best_only=True)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, epsilon=1e-4, mode='min')

    history = models.fit(x_train,
                         y_train,
                         epochs=num_epochs,
                         batch_size=200,
                         callbacks = [reduce_lr_loss, model_ckpt_callback],
                         validation_data=(x_val, y_val))
    loss, acc = models.evaluate(x_test, y_test, verbose=2)
    predicted_data = models.predict(x_test)
    accuracies_All.append([acc, key])
    print('accuracy =', acc)
    res = np.argmax(predicted_data, 1)
    y_test_res = np.argmax(y_test, 1)
    results = confusion_matrix((y_test_res+1), (res+1))
    confusion_matrices_All.append([results, key])