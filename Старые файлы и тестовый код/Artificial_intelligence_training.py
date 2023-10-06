import pickle as lpkl   # модуль Python, который предоставляет возможность сериализовать и десериализовать объекты Python
import itertools        # Python-модуль, который предоставляет набор функций для работы с итерируемыми объектами.
from random import shuffle  #  random () возвращает случайное число с плавающей точкой в промежутке от 0.0 до 1.0
from tensorflow.keras.datasets import cifar10
import os

# 2 кортежа:
# x_train, x_test: массив данных RGB изображений uint8 с формой (num_samples, 3, 32, 32) или (num_samples, 32, 32, 3), основанный на настройке бэкэнда image_data_format либо channels_first, либо channels_last соответственно.
# y_train, y_test: массив uint8 обозначений категорий с формой (num_samples, 1).

from tensorflow.keras import utils
# Какие-то утилиты

from tensorflow.keras import metrics
# Функция, которая используется для оценки работы вашей модели. Метрические функции предоставляются в параметре метрики при компиляции модели.

from tensorflow.keras.models import Sequential
# Функция, которая используется для оценки работы вашей модели. Метрические функции предоставляются в параметре метрики при компиляции модели.

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LSTM, BatchNormalization
#Dense реализует операцию: output = activation(dot(input, kernel) + bias), где активация — это функция активации по элементам, переданная в качестве аргумента активации, кернел — это матрица весов, созданная слоем, а смещение — это вектор смещения, созданный слоем (применимо только в случае, если use_bias — True).
# Замечание: если вход в слой имеет ранг больше 2, то он сглаживается перед исходным точечным продуктом с кернелом

# Dense Выравнивает вход. Не влияет на размер партии.

#Слой 2D свертки (например, пространственная свертка над изображениями).

# Conv2D Этот слой создает ядро свертки, которое свертывается со входом слоя для получения тензора выходов. Если значение параметра use_bias равно True, то создается и добавляется к выходным данным вектор смещения. Наконец, если активация не параметр None, он применяется и к выходным данным.
#При использовании этого слоя в качестве первого слоя в модели, задайте ключевой аргумент input_shape (кортеж целых чисел, не включает ось партии), например, input_shape=(128, 128, 3) для 128×128 RGB-картинок в data_format=»channels_last».

# MaxPooling2D Операция максимальной подвыборки(субдискретизации) для пространственных данных.

# LSTM Слой длинной кратковременной памяти — Hochreiter 1997.

# BatchNormalization  Слой пакетной нормализации (Иоффе и Сегеди, 2014).
# Нормализуйте активации предыдущего слоя в каждой партии, т.е. применяйте трансформацию, поддерживающую среднее значение активации близкое к 0, а стандартное отклонение активации близкое к 1.

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.layers import Reshape



def CNN():

    # Создание модели
    model = Sequential()

    # Входной слой Reshape
    model.add(Reshape((128, 6000, 1), input_shape=(128, 6000)))

    # Сверточный слой Conv2D
    model.add(Conv2D(256, (1, 3), activation='relu'))

    # Объединяющий слой MaxPooling2D
    model.add(MaxPooling2D((1, 2)))

    # Повторение сверточного слоя и объединяющего слоя
    model.add(Conv2D(256, (1, 3), activation='relu'))
    model.add(MaxPooling2D((1, 2)))

    model.add(Conv2D(256, (1, 3), activation='relu'))
    model.add(MaxPooling2D((1, 2)))

    # Слой Flatten
    model.add(Flatten())

    # Полносвязные слои Dense
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))

    # Выходной слой Dense
    model.add(Dense(8, activation='softmax'))

    # Компиляция модели
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Вывод структуры модели
    model.summary()
    return model


with open('/home/pov/.venv/intelintel/Qualification_Work/RDML/dataset', 'rb') as file:  # открываем файл 
    data = lpkl.load(file, encoding='Latin')                                          # Сохраняем данные в переменную

with open('/home/pov/.venv/intelintel/Qualification_Work/RDML/dataset', 'rb') as file:  # открываем файл 
    SNRs = lpkl.load(file, encoding='Latin')  

for key in data.keys():                                                                 # Перемешивание послдовательности случайным образом
    shuffle(data[key])       

new_data = {'комбинированный': []}              # Объявляется список
SNR_test = {}                                   # Объявляется список

for key in data.keys():                         # Перебор ключей
    train_len = int(0.9 * len(data[key]))       # len() возвращает длину (количество элементов) в объекте
                                                # int() округляет до целочисленного в меньшую сторону
    new_data['комбинированный'].append(data[key][:train_len])  # добавляет в конец списка
    SNR_test[key] = data[key][train_len:]   

new_data['комбинированный'] = list(itertools.chain.from_iterable(new_data['комбинированный']))  

outfile = open('/home/pov/.venv/intelintel/Qualification_Work/RDML/тестовые_образцы_ОСШ_новой_модели', 'wb')
lpkl.dump(SNR_test, outfile)
outfile.close()

outfile = open('/home/pov/.venv/intelintel/Qualification_Work/RDML/объединенные_данные_ОСШ', 'wb')
lpkl.dump(new_data, outfile)
outfile.close()

# Используйте этот код только в том случае, если вы хотите сгенерировать 20 различных моделей, соответствующих 20 значениям SNR
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

    models = CNN()
    opt = Adam(learning_rate=0.0003)
    models.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    num_epochs = 300

    # Checkpoint for models
    ckpt_folder = "CNN_МОДЕЛЬ/"
    ckpt_file_path = 'CNN_модель_ключи{}'.format(key)
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    model_ckpt_callback = ModelCheckpoint(filepath=ckpt_folder+ckpt_file_path,monitor='val_loss', mode='min', save_best_only=True)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, epsilon=1e-4, mode='min')

    history = models.fit(x_train,y_train, epochs=num_epochs, batch_size=200, callbacks = [reduce_lr_loss, model_ckpt_callback], validation_data=(x_val, y_val))
    loss, acc = models.evaluate(x_test, y_test, verbose=2)
    predicted_data = models.predict(x_test)
    accuracies_All.append([acc, key])
    print('accuracy =', acc)
    res = np.argmax(predicted_data, 1)
    y_test_res = np.argmax(y_test, 1)
    results = confusion_matrix((y_test_res+1), (res+1))
    confusion_matrices_All.append([results, key])