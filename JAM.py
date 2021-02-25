from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np


def rbf_k_pca(X, gamma, n_compon):
    """
    Реализация ядерного PCA с РБФ в качестве ядра
    :param X:{NumPy ndarray}, форма = [n_samples, n_features]
    :param gamma: float
                    Настроенчый параметр ядра РБФ
    :param n_compon: int
                    Число возращаемых главных компонент
    :return: X_pc {NumPy ndarray }, форма = [n_samples, k_features]
    спроецированный набор
    """
    # Попарно вычислить квадратичные евклидовы расстояния
    # в наборе данных размера MxN
    sq_dists = pdist(X, 'sqeuclidean')

    # Попарно конвертировать расстояния в квадратную матрицу
    mat_sq_dists = squareform(sq_dists)

    # Вычислить симметричную матрицу ядра
    K = exp(-gamma * mat_sq_dists)

    # Центрировать матрицу ядра
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K)
    X_pc = np.column_stack((eigvecs[:, -i]
                            for i in range(1, n_compon + 1)))
    return X_pc


from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, random_state=123)
import matplotlib.pyplot as plt

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
plt.show()
from matplotlib.ticker import FormatStrFormatter

X_kpca = rbf_k_pca(X, gamma=15, n_compon=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1],
              color='blue', marker='o', alpha=0.5)
ax[1].scatter(X_kpca[y == 0, 0], np.zeros((50, 1)) + 0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros((50, 1)) - 0.02,
              color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=100, random_state=1, noise=0.1, factor=0.2)
plt.scatter(X[y == 0, 0], X[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1],
            color='blue', marker='o', alpha=0.5)
plt.show()