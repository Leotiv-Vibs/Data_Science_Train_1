from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def plot_dic(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ("s", "x", 'o', "^", "v")
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
    if test_idx:
        X_test, Y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=np.array([0.5, 0.5, 0.5]),
                    alpha=1.0, linewidths=1, marker='o',
                    s=55, label='тестовый набор')


iris = datasets.load_iris()
x = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
ppn = Perceptron(n_iter_no_change=40, eta0=0.3, random_state=0)
ppn.fit(X_train_std, Y_train)
y_pred = ppn.predict(X_test_std)
print('Число: %d' % (Y_test != y_pred).sum())
print('def: %.2f' % accuracy_score(Y_test, y_pred))

x_com_std = np.vstack((X_train_std, X_test_std))
y_com = np.hstack((Y_train, Y_test))
plot_dic(X=x_com_std, y=y_com, classifier=ppn, test_idx=range(105, 150))
plt.xlabel('длина чашелистника[]')
plt.ylabel('длина лепестка[стадантизированная]')
plt.legend(loc='upper left')
plt.show()
