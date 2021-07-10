import xlrd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from pandas.core.frame import DataFrame


# 读取数据
def read_xlsx(csv_path):
    data = pd.read_csv(csv_path)
    print(data)
    return data


def random_sampling(x,y, m):
    x = np.array(x)
    y = np.array(y)
    a = np.random.permutation(len(x))
    subset = x[a]
    label = y[a]
    return subset, label

def model(x_train,y_train,x_test):

    model = DecisionTreeClassifier()
    yclass = []
    for i in range(20):
        subset, label = random_sampling(x_train, y_train, 150)
        model.fit(subset, label)
        a = model.predict(x_test)
        a = list(a)
        yclass.append(a)
    data = DataFrame(yclass)
    ypred = []
    for col in data.columns:
        mean = data[col].mean()
        ypred.append(mean)
    print(ypred)
    return ypred



def accuracy(ypred, y_test):
    correct = 0
    y_test = list(y_test)
    for x in range(len(y_test)):
        if ypred[x] == y_test[x]:
            correct += 1
    accuracy = (correct / float(len(y_test))) * 100.0
    print("Accuracy:", accuracy, "%")
    return accuracy


if __name__ == '__main__':
    data = read_xlsx(r'D:\数据集\zzl_data\win.csv')
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    random_sampling(x_train, y_train, 150)
    ypred = model(x_train,y_train,x_test)
    # x = data.iloc[:, :-1]
    # y = data.iloc[:, -1]
    # x_train, x_test, y_train, y_test = train_test_split(x, y)
    # labels = voting(x_train, x_test, y_train)
    accuracy(ypred, y_test)






