# Среда Python 3 поставляется с множеством установленных полезных библиотек для аналитики
# Он определяется Docker-образом kaggle/python: https://github.com/kaggle/docker-python
# Например, вот несколько полезных пакетов для загрузки

import numpy as np # линейная алгебра
import pandas as pd # обработка данных, ввод/вывод CSV-файлов (например, pd.read_csv)

# Файлы входных данных доступны в директории "../input/", доступной только для чтения
# Например, запустив эту программу (нажав кнопку run или Shift+Enter), вы получите список всех файлов в каталоге ввода

import os
for dirname, _, filenames in os.walk('/home/pov/.venv/intelintel/Qualification_Work/RDML'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# В текущий каталог (/home/pov/.venv/intelintel/Qualification_Work/RDML) можно записать до 20 ГБ, которые будут сохранены в качестве выходных данных при создании версии с помощью "Save & Run All". 
# Можно также записывать временные файлы в каталог /kaggle/temp/, но они не будут сохранены за пределами текущей сессии

# загрузчик данных
import pickle
# Pickle – это модуль Python, который предоставляет возможность сериализовать и десериализовать объекты Python. Сериализация – это процесс преобразования объекта в поток байтов, который затем может быть сохранен в файл или передан через сеть.
import gzip
# Этот модуль предоставляет простой интерфейс для сжатия и распаковки файлов подобно тому, как это делают GNU-программы gzip и gunzip.
import numpy as np
# Часть кода import numpy указывает Python на необходимость привнести библиотеку NumPy в текущее окружение.
# Часть кода as np указывает Python на присвоение NumPy псевдонима np. Это позволит вам использовать функции NumPy, просто набирая np.function_name, а не numpy.function_name.

data_path = "/home/pov/.venv/intelintel/Qualification_Work/RDML/RML2016.10a_dict.pkl"

with open(data_path, 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()

    m_set = set()
s_set = set()
p_keys = list(p.keys())
for modulation, snr in p_keys:
    m_set.add(modulation)
    s_set.add(snr) 

# Кол-во и модуляции
print(len(m_set), m_set)
# Кол-во уровней ОСШ и уровни ОСШ
print(len(s_set), s_set)

# выбрать один тип
print("type:",p_keys[1])
p_t1 = p[p_keys[1]]
print("образцы одного типа:", len(p_t1))
print("компоненты в одном образце:", len(p_t1[0]))
print("димондиональность компонента", len(p_t1[0][0]), len(p_t1[0][1]))

print("общее количество образцов:", len(p_keys) * len(p_t1))

import pandas as pd
pd.concat([
    pd.DataFrame(p_t1[0][0], columns=["in-phase"]).describe(),
    pd.DataFrame(p_t1[0][0], columns=["quadrature"]).describe()
], axis=1)