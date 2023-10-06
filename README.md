# Qualification_Work
Для запуска следует скачать имеющийся пресет данных
https://mega.nz/file/h3cXiZKa#ptehZ2QULEv1GbRKQ26QVjDAMvrjBQGssXYLz1mbSWE
https://mega.nz/file/tqsTXCra#QFrrYBi-UoK01tt2yeRAy9DJ322b6T2kwAQpQnRCuY4
И поместить их в папку /RDML

Настройка виртуальной машины venv. Необходимость виртуальной машины обусловлена тем, что последние версии Linux систем требуют разделение пакетного пространства для apt и pip.

sudo apt update
sudo apt-get -y install pip python3-venv build-essential libssl-dev libffi-dev python3-dev
venv (По умолчанию уже установлен)

python -m venv /home/$USER/.venv/intelintel Создается путь с исполняемыми файлами

Установка библиотек:

/home/$USER/.venv/intelintel/bin/pip install tensorflow \n
/home/$USER/.venv/intelintel/bin/pip install keras  \n
/home/$USER/.venv/intelintel/bin/pip install matplotlib \n
/home/$USER/.venv/intelintel/bin/pip install pandas \n
/home/$USER/.venv/intelintel/bin/pip install metrics    \n
/home/$USER/.venv/intelintel/bin/pip install cifar10    \n
/home/$USER/.venv/intelintel/bin/pip install scikit-learn   \n
/home/$USER/.venv/intelintel/bin/pip install theano \n
/home/$USER/.venv/intelintel/bin/pip install --upgrade theano \n
/home/$USER/.venv/intelintel/bin/pip install -c mila-udem -c mila-udem/label/pre theano pygpu \n
/home/$USER/.venv/intelintel/bin/pip install seaborn \n
/home/$USER/.venv/intelintel/bin/pip install --upgrade PyQt5 \n

Используемый редактор кода (IDE):

VS code

Установленные расширения: 

GPU Environments v0.3.0
Pylance v2023.9.20
Python v2023.16.0
Python Auto Venv v1.3.2 (для использования виртуальной машины)
Python Environment Manager v1.2.4
Jupyter
Jupyter Cell Tags
Jupyter Keymap
Jupyter Notebook
Jupyter Slide Show
Так же установлены расширения для русификации


Система: 

Linux MX 23, окружение KDE, основан на Debian 12.2.0-14
Версия ядра 6.1.0-10-amd64

Железо: 

Intel Core I3 7100
Nvidia 940MX


Настройка если драйверов CUDA была осуществлена корректно (CuDNN и т. п. драйвера), то расчет должен пойти на них. Имеющееся количество эпох - 100, на заданной системе расчет одной эпохи занимает 12-13 минут.

