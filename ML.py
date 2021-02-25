import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_dic(X, y, classifier, resolution=0.02):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)


class Perceptron(object):
    """
    Классификатор на основе пересептронаю.
    Параметры
    __________
    eta : float
        Темп обучения между(0.0 - 1.0)
    n_iter : int
        Проходы по тренировочному набору данных.

    Атрибуты
    __________
    w_ : 1 - мерный массив
        Весовые коэфиценты после подгонки
    errors_ : список
        Числа случаев ошибочной классификации в каждой точке
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x2, y2):
        self.w_ = np.zeros(1 + x2.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x2, y2):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
df.tail()

y = df.iloc[0:100, 4].values
y1 = np.where(y == 'Iris-setosa', -1, 1)
x1 = df.iloc[0:100, [0, 2]].values
plt.scatter(x1[:50, 0], x1[:50, 1],
            color='red', marker='o', label='щетинистый')
plt.scatter(x1[50:100, 0], x1[50:100, 1],
            color='blue', marker='x', label='разноцветный'
            )
plt.xlabel('длина чашелистника')
plt.ylabel('длина лепестка')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(eta=0.3, n_iter=100)
ppn.fit(x1, y1)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Число ошибочных случаев')
plt.show()

plot_dic(x1, y1, classifier=ppn)
plt.xlabel('Длина чашелистника[см]')
plt.ylabel('Длина лепестка[см]')
plt.legend(loc='upper left')
plt.show()
