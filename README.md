# Qualification_Work
Система: 

Linux MX 23, окружение KDE, основан на Debian 12.2.0-14
Версия ядра 6.1.0-10-amd64

Железо: 

Intel Core I3 7100
Nvidia 940MX

Используемый редактор кода (IDE):

VS code

Установленные расширения: 

GPU Environments v0.3.0
Pylance v2023.9.20
Python v2023.16.0
Python Auto Venv v1.3.2 (для использования виртуальной машины)
Python Environment Manager v1.2.4

Так же установлены расширения для русификации

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



Настройка драйверов CUDA не была осуществлена корректно (установлены все драйвера cuDNN, CUDA, tensorflow-gpu автоматически встроена в библиотеку tensorflow, однако библиотека в тупую не видит чип)

GitFlic init:

cd ~/home/$USER/.venv/intelintel/Qualification_Work
git config --global user.name "paxant"
git config --global user.email "s.iuzer2015@yandex.ru"

git init
touch README.md^C
git add README.md
git commit -m "add README"
git add --all
git commit -m "add files"
git remote add origin https://gitflic.ru/project/paxant/qualification_work.git
git push -u origin master или git push origin master --force

Username for 'https://gitflic.ru': paxant
Password:            1

Для запуска кода необходимо в VS code открыть содержимое папки /inteintel, для исключения ошибок, связанных с подключением библиотек

При создании проекта сначала требуется запустить файл RadioML_2016_encode для преобразования имеющегося файла в удобоваримый вид# Qualification_Work
