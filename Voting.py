
import xlrd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

#读取数据
def read_xlsx(csv_path):
    data = pd.read_csv(csv_path)
    print(data)
    return data

#最大投票法
def voting(x_train,x_test,y_train):
    model1 = DecisionTreeClassifier()
    model2 = KNeighborsClassifier()
    model3 = MultinomialNB()
    model1.fit(x_train,y_train)
    model2.fit(x_train,y_train)
    model3.fit(x_train,y_train)

    a = model1.predict(x_test)
    b = model2.predict(x_test)
    c = model3.predict(x_test)
    labels = []
    for i in range(len(x_test)):
        ypred = []
        ypred.append(a[i])
        ypred.append(b[i])
        ypred.append(c[i])
        counts = np.bincount(ypred)
        label = np.argmax(counts)
        labels.append(label)
    print(labels)
    return labels

def accuracy(labels, y_test):
    correct = 0
    y_test = list(y_test)
    for x in range(len(y_test)):
        if labels[x] == y_test[x]:
           correct += 1
    accuracy = (correct / float(len(y_test))) * 100.0
    print("Accuracy:", accuracy, "%")
    return accuracy

if __name__ == '__main__':
    data = read_xlsx(r'D:\数据集\zzl_data\win.csv')
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    labels = voting(x_train,x_test,y_train)
    accuracy(labels, y_test)






