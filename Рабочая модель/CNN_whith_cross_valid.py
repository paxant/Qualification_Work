# Импорт библиотек для работы с файлами и т.д.
import os, random
import pickle as lpkl
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import confusion_matrix
import pickle as cPickle
import seaborn as sns
# Импорт библиотек для ИИ
import tensorflow as tf
import tensorflow.keras.models as models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import utils
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Reshape, Flatten,Activation,Dropout ,Conv2D, MaxPooling2D, ZeroPadding2D, LSTM, BatchNormalization, GaussianNoise
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import *
from keras.utils import plot_model
from sklearn.model_selection import StratifiedKFold

Project_Dir = '/home/pov/.venv/intelintel/Qualification_Work/RDML/'

Signal_name = []
Signals_Dir = Project_Dir + 'Signals'
print(f"Файл с названиями сигналов существует: {os.path.isfile(Signals_Dir)}")

if os.path.isfile(Signals_Dir) == False:
    print("Сначала запустите файл RadioML_2016_encode.py")
    exit ()
Signal = open(Signals_Dir, 'rb')
Signal_name = lpkl.load(Signal)
Signal.close 

Signal_name.remove('AM-DSB')    # Удаляем все, что не используем из списка
# Signal_name.remove('AM-SSB')
Signal_name.remove('WBFM')

Signal_name = sorted(Signal_name)

for i in range(len(Signal_name)):
    print(f" {Signal_name[i]} файл с названиями сигналов существует: {os.path.isfile('/home/pov/.venv/intelintel/Qualification_Work/RDML/'+Signal_name[i])}")

DATASET = {}
SNR_data = []

for i in range(len(Signal_name)):                                         # создание словаря классов для преобразования строк в цифры.
    infile = open(Project_Dir + Signal_name[i],'rb')      # Вывод в файл датасета
    Signal_Range = lpkl.load(infile)
    SNR = -20
    for j in range(20):             #До 18 дБ ОСШ
        if len(SNR_data) != 20: 
            SNR_data.append(SNR)
        DATASET[(Signal_name[i], SNR)] = Signal_Range[j]
        SNR = SNR + 2


#with open("/home/pov/.venv/intelintel/Qualification_Work/RDML/RML2016.10a_dict.pkl", 'rb') as f:
#    Xd = cPickle.load(f, encoding="latin1") 
 
#snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
# snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], DATASET.keys())))), [1,0])
 
# map() принимает функцию и итерацию (или несколько итераций) в качестве аргументов и возвращает итератор, 
# который выдает преобразованные элементы по запросу. Сигнатура функции map определяется следующим образом:
# map(function, iterable[, iterable1, iterable2,..., iterableN])

# Лямбда принимает любое количество аргументов (или ни одного), но состоит из одного выражения.
# Возвращаемое значение — значение, которому присвоена функция. 

# Функция sorted() возвращает новый отсортированный список итерируемого объекта (списка, словаря, кортежа). 
# По умолчанию она сортирует его по возрастанию.

# Функция set разбивает строку на символы, а из символов формирует множество.

# Таким образом данная строчка получает сортированный список ключей, однако, они были получены ранее

X = []                                  # Создание списка
lbl = []                                # Создание списка
for mod in Signal_name:
    for snr in SNR_data:
        X.append(DATASET[(mod,snr)])
        for i in range(DATASET[(mod,snr)].shape[0]):  
            lbl.append((mod,snr))
           # print(i)
# Данный цикл перебирает все значения Датасета в новый формат

X = np.vstack(X)                    # Возвращает вертикальное значение массива
np.random.seed(2016)   

# Функция numpy.random.seed() используется для установки начального числа для алгоритма генератора
#  псевдослучайных чисел в Python.
# Это удобная устаревшая функция, которая существует для поддержки старого кода, 
# использующего синглтон RandomState. Лучше всего использовать специальный экземпляр 
# генератора, а не методы генерации случайных величин, представленные непосредственно 
# в модуле случайных чисел

n_examples = int(X.shape[0])            # общее кол-во значений

n_train = int(n_examples * 0.8)         # Кол-во выборок для обучения
n_valid = int(n_examples * 0.1)         # Кол-во выборок для контроля
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)  # Выборка
valid_idx = np.random.choice(range(0,n_examples), size=n_valid, replace=False)  
test_idx = list(set(range(0,n_examples))-set(train_idx)-set(valid_idx))
#test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
X_valid = X[valid_idx]              # Распределение выборок

def to_onehot(yy):
    data = list(yy)
    yy1 = np.zeros([len(data), max(data)+1])
    yy1[np.arange(len(data)),data] = 1
    return yy1

# max() находит максимальное значение среди последовательности
# len() длина последовательности 
# np.zeros() возвращает новый массив заданной формы и типа, где значение элемента равно 0
# np.arange() возвращает объект типа ndarray с равномерно расположенными значениями внутри
# заданного интервала 

print(lbl[0][0])
Y_train = to_onehot(map(lambda x: Signal_name.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: Signal_name.index(lbl[x][0]), test_idx))
Y_valid = to_onehot(map(lambda x: Signal_name.index(lbl[x][0]), valid_idx))

# полечаем последовательности для обучения, проверки и теста

in_shp = list(X_train.shape[1:])

# .shape возвращает структуру массива, его форму, размерность
# [1:] адрессует к первому элементу функции shape
# Функция Python list() принимает в качестве параметра любой итерируемый объект
# и возвращает список. В Python iterable - это объект, над которым можно выполнять
# итерацию. Примерами итерируемых объектов являются кортежи, строки и списки.

print(X_train.shape, in_shp)
classes = Signal_name
print(classes)
print(SNR)

for i in range(8):
    print('Номер класса: ', np.argmax(Y_train[i]))
    print('Имя класса: ', classes[np.argmax(Y_train[i])])
    plt.figure(figsize=(800/96, 800/96), dpi=96)
# используется в Python для получения индексов максимального 
# элемента из массива (одномерный массив) или любой строки или столбца 
# (многомерный массив) любого заданного массива.
    plt.xlabel("")
    plt.ylabel("")
    plt.title(classes[np.argmax(Y_train[i])])
    plt.imshow(X_train[i])
    plt.show()

def CNN():
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
    model.summary()
    return model

def ReLU(x):
    """Считываемая функция активации ReLU"""
    if x < 0:
        return 0
    else:
        return x

plt.plot(np.arange(-10, 10), [ReLU(x) for x in np.arange(-10, 10)])
plt.title('ReLU')
plt.show()

def sigmoid(x):
    """-> 0,1 = ВЕРОЯТНОСТЬ"""
    return 1 / (1 + np.exp(-x))

plt.plot(np.arange(-10, 10), [sigmoid(x) for x in np.arange(-10, 10)])
plt.title('Sigmoid')
plt.show()

def softmax(x):
    """Вычислить значения softmax для каждого набора оценок в x.
    
    -> 0,1 = ВЕРОЯТНОСТЬ"""
    return np.exp(x) / np.sum(np.exp(X))

x = np.arange(-10, 10)

a = softmax(x)
print(a)

model = CNN()


# monitor: Количество, подлежащее мониторингу.

# min_delta: Минимальное изменение контролируемого количества, которое будет считаться 
# улучшением, т.е. абсолютное изменение меньше min_delta будет считаться отсутствием улучшения.

# patience (терпение): Количество эпох без улучшения, после которого обучение будет остановлено.

# verbose: Режим вербозности, 0 или 1. В режиме 0 ничего не говорится, а в режиме 1 сообщения
# выводятся, когда обратный вызов выполняет какое-либо действие.

# mode (режим): Одно из {"auto", "min", "max"}. В режиме "min" обучение остановится, 
# когда контролируемая величина перестанет уменьшаться, в режиме "max" - когда контролируемая
# величина перестанет увеличиваться, в режиме "auto" направление автоматически определяется по 
# имени контролируемой величины.

# baseline (базовая линия): Базовое значение для контролируемой величины. Обучение будет 
# прекращено, если модель не покажет улучшения по сравнению с базовым значением.

# restore_best_weights: Восстанавливать ли веса модели из эпохи с наилучшим значением 
# контролируемой величины. Если False, то используются веса модели, полученные на последнем 
# шаге обучения. Эпоха будет восстановлена независимо от производительности относительно базовой 
# линии. Если ни одна эпоха не улучшает базовую, то обучение будет проводиться на терпеливых 
# эпохах с восстановлением весов из лучшей эпохи в этом наборе.

# start_from_epoch: Количество эпох, которое необходимо выждать, прежде чем начать отслеживать
# улучшение. Это позволяет сделать период разминки, в течение которого улучшения не ожидается и, 
# следовательно, обучение не будет остановлено.

accuracies_All = []
confusion_matrices_All = []

filepath = Project_Dir + 'RDML2016B_обучуенная_модель_2.wts.h5'
model_ckpt_callback = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, mode='auto', save_best_only=True)
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', patience=15, verbose=1, mode='auto')
batch_size=200
nb_epoch = 150
acc_per_fold = []
loss_per_fold = []
plot_model(model, to_file=Project_Dir+'model.png')
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=78)
scores = []
for train, test in kfold.split(X_train, Y_train):
    fold_no = 1
    history = model.fit(X_train[train], Y_train[train], epochs=nb_epoch, batch_size=batch_size, callbacks = [reduce_lr_loss, model_ckpt_callback])
    scores = model.evaluate(X_train[test], Y_train[test], verbose=1)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])   
    fold_no = fold_no + 1
model.load_weights(filepath)

# Показать простую версию исполнения
score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
print(score)

plt.figure()
plt.title('Эффективность обучения')
plt.plot(history.epoch, history.history['loss'], label='потери в тренировке+ошибка')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.show()


loss, acc = model.evaluate(X_test, Y_test, verbose=1)
predicted_data = model.predict(X_test)
accuracies_All.append([acc, SNR_data])
print('точность =', acc)
res = np.argmax(predicted_data, 1)
y_test_res = np.argmax(Y_test, 1)
results = confusion_matrix((y_test_res+1), (res+1))
confusion_matrices_All.append([results, SNR_data]) 

# насчет кросс-валидации сказать сложно, перемешивание данных произошло, однако про степень
# кросс-валидации сказать ничего не могу, пока не разобрался

def plot_confusion_matrix(cm, title='Матрица запутанности', cmap=plt.cm.Blues, labels=[]):
    my_dpi=96
    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('Истинная метка')
    plt.xlabel('Прогнозируемая метка')
    plt.show()

test_Y_hat = model.predict(X_test, batch_size=batch_size, verbose = 1)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)

# Построение матрицы смешения
acc = {}
for snr in SNR_data:

    # извлечение классов @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    
 
    # оценочные классы
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plot_confusion_matrix(confnorm, labels=classes, title="Матрица смешения ConvNet (SNR=%d)"%(snr))

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Общая точность: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)

# Сохранение результатов в pickle-файле для последующего построения графиков
print(acc)


# Построение кривой точности
plt.plot(SNR_data, list(map(lambda x: acc[x], SNR_data)))
plt.xlabel("Отношение сигнал/шум")
plt.ylabel("Точность классификации")
plt.title("Точность классификации CNN2 на RadioML 2016.10b")
plt.show()
import pandas