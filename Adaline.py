import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from numpy.random import seed


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


class Adaline(object):
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
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(x2)
            errors = (y2 - output)
            self.w_[1:] += self.eta * x2.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, x):
        return self.net_input(x)

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
df.tail()

y = df.iloc[0:100, 4].values
y1 = np.where(y == 'Iris-setosa', -1, 1)
x1 = df.iloc[0:100, [0, 2]].values

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ada1 = Adaline(n_iter=10, eta=0.01).fit(x1, y1)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log(ada1.cost_), marker='o')
ax[0].set_xlabel('Эпохи')
ax[0].set_ylabel('log(Сумма квадратичных ошибок)')
ax[0].set_title('Adaline(темп обучения 0.01)')
ada2 = Adaline(n_iter=10, eta=0.0001).fit(x1, y1)
ax[1].plot(range(1, len(ada2.cost_) + 1), np.log(ada2.cost_), marker='o')
ax[1].set_xlabel('Эпохи')
ax[1].set_ylabel('log(Сумма квадратичных ошибок)')
ax[1].set_title('Adaline(темп обучения 0.0001)')
plt.show()

x_std = np.copy(x1)
x_std[:, 0] = (x1[:, 0] - x1[:, 0].mean()) / x1[:, 0].std()
x_std[:, 1] = (x1[:, 1] - x1[:, 1].mean()) / x1[:, 1].std()

ada = Adaline(n_iter=15, eta=0.01)
ada.fit(x_std, y1)
plot_dic(x_std, y1, classifier=ada)
plt.title("Adaline(градиентный спуск)")
plt.xlabel('длина чашелистника[]')
plt.ylabel('длина лепестка[стадантизированная]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Эпоих')
plt.ylabel('Сумма квадратчных ошибок')
plt.show()


class AdalineSGD(object):
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

    def __init__(self, eta=0.01, n_iter=10, shufle=True, random_state=None):
        self.eta = eta
        self.w_init = False
        self.shufle = shufle
        self.n_iter = n_iter
        if random_state:
            seed(random_state)

    def fit(self, x2, y2):
        self._init_w(x2.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shufle:
                x2, y2 = self._shufle(x2, y2)
            cost = []
            for xi, target in zip(x2, y2):
                cost.append(self._update_w(xi, target))
            avg_cost = sum(cost) / len(y2)
            self.cost_.append(avg_cost)
        return self

    def partical_fit(self, x, y):
        if not self.w_init:
            self._init_w(x.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self._update_w(xi, target)
        else:
            self._update_w(x, y)
        return self

    def _shufle(self, x, y):
        r = np.random.permutation(len(y))
        return x[r], y[r]

    def _init_w(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_init = True

    def _update_w(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, x):
        return self.net_input(x)

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)


ADA = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ADA.fit(x_std, y1)
plot_dic(x_std, y1, classifier=ADA)
plt.title('Adaline(стохастический градиентный спуск)')
plt.xlabel('длина чашелистника[]')
plt.ylabel('длина лепестка[стадантизированная]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ADA.cost_) + 1), ADA.cost_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Средняя стоимость')
plt.show()
