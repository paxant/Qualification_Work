# https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb
# Далее будет использоваться датасет RadioML2016

# Для чтения pkl файлов импортируем библиотеку pickle

# В дальнейшем
# () - кортеж
# Кортеж - это последовательность элементов, которые не могут быть изменены (неизменяемы).
# [] - список
# Список - это последовательность элементов, которые могут быть изменены (изменяемые).
# {} - словарь или набор


import pickle as lpkl                       # Обозначим pickle как library pickle (сокращено lpkl)
import os                                   # Поключение модуля для работы с операционной системой
import numpy as np
import csv
import pickle as cPickle
def digitizer(labels):
    
    unique_labels = np.unique(labels)       # находит уникальные элементы массива и возвращает отсортированные уникальные элементы входного массива и, при необходимости, возвращает индексы входного массива, которые дают уникальные значения и восстанавливают входной массив.
    label_dict = {}                         # Создаем список
    num = 1                                 # Счетчик
    for i in unique_labels:                 # Перебор полученных значений в unique labels 
        label_dict[i] = num                 # Задаем метку для полученного списка манипуляций
        num += 1                            # + 1 к счетчику

    digit_label = []                        # Создаем список
    for i in labels:                        # Перебор значний изначального списка
        digit_label.append(label_dict[i])   # Сохраняем номера манипуляций в их представленном порядке в переменной labels
    
    return label_dict,  digit_label         #Возвращаем переменные

def onehot_encoder(L_dict,  labels):        # Функция какой-то горячий кодеровщик
    num_classes = len(L_dict)               #
    vector = np.zeros(num_classes)          #
    vector[L_dict[labels]-1] = 1            #
    return vector                           #

print(os.name)

#Radio_ML_dirrectory = '/home/pov/.venv/intelintel/Qualification_Work/RDML/RML2016.10a_dict.pkl';
#print(f"Этот файл существует: {os.path.isfile(Radio_ML_dirrectory)}")

#with open(Radio_ML_dirrectory, 'rb') as file:       # with автоматически закроет файл при выходе из блока
   # Data_Sample = lpkl._Unpickler(file)             # Для чтение двоичного потока из файла
   # Data_Sample.encoding = 'latin1'                 # Декодирование из 8-ми битной кодировки с поддержкой ASCII
    #_Data_Sample = Data_Sample.load()

Read_RDML_b = '/home/pov/.venv/intelintel/Qualification_Work/RDML/RML2016.10b.dat'
print(f"Этот файл существует: {os.path.isfile(Read_RDML_b)}")

if os.path.isfile(Read_RDML_b) == False:
    print("Настройте дирректорию, либо скачайте файл с данными")
    print("https://mega.nz/file/tqsTXCra#QFrrYBi-UoK01tt2yeRAy9DJ322b6T2kwAQpQnRCuY4")
    print("https://mega.nz/file/h3cXiZKa#ptehZ2QULEv1GbRKQ26QVjDAMvrjBQGssXYLz1mbSWE")
    exit ()

_Data_Sample = cPickle.load(open(Read_RDML_b,'rb'), encoding='latin')

    

print(_Data_Sample[('8PSK', -12)])
# snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], _Data_Sample.keys())))), [1,0])
# Присвоить переменным snsr и mods. Lambda некий тип с ограниченными возможностями.  
# Функция sorted() возвращает новый отсортированный список итерируемого объекта (списка, словаря, кортежа).
# По умолчанию она сортирует его по возрастанию.
# В Python есть ключевое слово list ().
# Это функция, которая либо создает пустой список, либо приводит к списку итерируемый объект.
# Python map() — это встроенная функция, которая позволяет обрабатывать и преобразовывать все элементы в 
# итерируемом объекте без использования явного цикла for, методом, широко известным как сопоставление 
# (mapping). map() полезен, когда вам нужно применить функцию преобразования к каждому элементу в коллекции 
# или в массиве и преобразовать их в новый массив.
# тот метод также возвращает итерируемый объект. Он является списком всех ключей в словаре.
# Как и метод items(), этот отображает изменения в самом словаре.

Array = []                                      # Создаем массив для работы с данными файлов
for i in _Data_Sample.keys():   
    if i[0] not in Array:
        Array.append(i[0])                      # append() добавляет в конец списка элемент, переданный ему в качестве аргумента.

outfile = open('/home/pov/.venv/intelintel/Qualification_Work/RDML/'+'Signals','wb')      # Вывод в файл датасета
lpkl.dump(Array,outfile)                       # Байтовая запись в файл
outfile.close() 

Datas = []
for i in range(len(Array)):                                         # создание словаря классов для преобразования строк в цифры.
    for j in range(-20, 20, 2):
        Datas.append(_Data_Sample[(Array[i], j)])
    outfile = open('/home/pov/.venv/intelintel/Qualification_Work/RDML/'+Array[i],'wb')      # Вывод в файл датасета
    lpkl.dump(Datas,outfile)                       # Байтовая запись в файл
    outfile.close() 
    Datas.clear()

SNRs = {}                                       # Получим список SNR 
for key in _Data_Sample.keys():                 # 
    if key not in SNRs:                         # Если такого ключа нет, то он добавляется в список
        SNRs[key[1]] = []
                          
outfile = open('/home/pov/.venv/intelintel/Qualification_Work/RDML/'+'SNRs','wb')
lpkl.dump(Datas,outfile)                       # Байтовая запись в файл
outfile.close() 

