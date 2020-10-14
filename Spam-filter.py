#Импортируем библиотеки
import glob
import numpy as np
import re
from random import shuffle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

#Списки путей к ham и spam
spam_path = glob.glob('C:/Users/rusla/Desktop/enron/enron*/spam/*.txt')
ham_path = glob.glob('C:/Users/rusla/Desktop/enron/enron*/ham/*.txt')

#Перемешиваем списки путей
shuffle(spam_path)
shuffle(ham_path)

#80% файлов используем под обучение, остальное под тестирование
spam_documents = np.int32(len(spam_path)*0.8)
ham_documents = np.int32(len(ham_path)*0.8)
R = spam_documents + ham_documents #Общее кол-во документов

#Списки путей к ham и spam для обучения и тестирования
spam_train = spam_path[:spam_documents]
ham_train = ham_path[:ham_documents]
spam_test = spam_path[spam_documents:]
ham_test = ham_path[ham_documents:]

#Функция обработки текста
def text_editor(text):
    text = text.lower() #Заменяем все буквы на строчные
    text = re.sub(r'[0123456789?.,;:!~`@"№#№$%^&*()-_+=<>/|\s]',' ', text) #Убираем незначащие символы
    text = re.sub(r'\s+',' ', text) #Убираем лишние пробелы
    split_text = text.split() #Создаем массив слов
    #Убираем шумовые слова
    stopWords = set(stopwords.words('english'))
    for word in split_text:
        if len(word) < 3 or word in stopWords:
            split_text.remove(word)
    return split_text
            
#Словари для spam и ham
spam_dictionary = {}
ham_dictionary = {}

#Заполняем словари для spam и ham
for file in spam_train:
    with open(file, errors = 'ignore') as f:#Работаем с файлом игнорируя исключительные ситуации
        text = f.read()#Cчитываем текст в строку
        text = text_editor(text)#Преобразовываем текст
        for word in text:
            if word in spam_dictionary:#Если слово встречалось в словаре, увеличиваем их кол-во
                spam_dictionary[word] += 1
            else:#Если не встречалось, присваиваем единицу
                spam_dictionary[word] = 1
                
for file in ham_train:
    with open(file, errors = 'ignore') as f:#Работаем с файлом игнорируя исключительные ситуации
        text = f.read()#Cчитываем текст в строку
        text = text_editor(text)#Преобразовываем текст
        for word in text:
            if word in ham_dictionary:#Если слово встречалось в словаре, увеличиваем их кол-во
                ham_dictionary[word] += 1
            else:#Если не встречалось, присваиваем единицу
                ham_dictionary[word] = 1

#Словарь уникальных слов
dictionary = set(spam_dictionary)
dictionary.update(set(ham_dictionary))
v = len(dictionary)

#Вероятности класса С0 и С1
ProbC0 = np.log(ham_documents/R)
ProbC1 = np.log(spam_documents/R)

#Метрики
TP = 0
TN = 0
FP = 0
FN = 0

#Тестирование
for file in ham_test:
    C0 = ProbC0
    C1 = ProbC1
    with open(file, errors = 'ignore') as f:#Работаем с файлом игнорируя исключительные ситуации
        text = f.read()#Cчитываем текст в строку
        text = text_editor(text)#Преобразовываем текст
        for word in text:
            if word in ham_dictionary.keys():              
                C0 += (np.log(ham_dictionary.get(word) + 1) - np.log(v + sum(ham_dictionary.values()))) 
            else:
                C0 += (np.log(1) - np.log(v + sum(ham_dictionary.values()))) 
            if word in spam_dictionary.keys():
                C1 += (np.log(spam_dictionary.get(word) + 1) - np.log(v + sum(spam_dictionary.values()))) 
            else:
                C1 += (np.log(1) - np.log(v + sum(spam_dictionary.values())))
    if C0 > C1:
        TN += 1
    else:
        FP += 1
        
for file in spam_test:
    C0 = ProbC0
    C1 = ProbC1
    with open(file, errors = 'ignore') as f:#Работаем с файлом игнорируя исключительные ситуации
        text = f.read()#Cчитываем текст в строку
        text = text_editor(text)#Преобразовываем текст
        for word in text:
            if word in ham_dictionary.keys():              
                C0 += (np.log(ham_dictionary.get(word) + 1) - np.log(v + sum(ham_dictionary.values()))) 
            else:
                C0 += (np.log(1) - np.log(v + sum(ham_dictionary.values()))) 
            if word in spam_dictionary.keys():
                C1 += (np.log(spam_dictionary.get(word) + 1) - np.log(v + sum(spam_dictionary.values()))) 
            else:
                C1 += (np.log(1) - np.log(v + sum(spam_dictionary.values())))
    if C0 > C1:
        FN += 1
    else:
        TP += 1

#Высчитываем метрики
accuracy = (TP + TN)/(TP + FN + TN + FP)
a = FP/(FP + TN)
b = FN/(TP + FN)
precision = TP/(TP + FP)
recall = TP/(TP + FN)
print(accuracy, a, b, precision, recall) #Печатаем метрики