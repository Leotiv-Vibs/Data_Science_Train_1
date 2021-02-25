import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df_wine = pd.read_csv(url, header=None)

x1, y1 = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=0)
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)

cov_mat = np.cov(X_train_std.T)

np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print('BC %s: %s\n' % (label, mean_vecs[label - 1]))

d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print('Внутриклассовая матрица разброса: %sx%s ' % (S_W.shape[0], S_W.shape[1]))
mean_oweral = np.mean(X_train_std, axis=0)
d = 13
S_B = np.zeros((d, d))
for i, mean_vecs in enumerate(mean_vecs):
    n = X_train[y_train == i + 1, :].shape[0]
    mean_vecs = mean_vecs.reshape(d, 1)
    mean_oweral = mean_oweral.reshape(d, 1)
S_B += n * (mean_vecs - mean_oweral).dot((mean_vecs - mean_oweral).T)
print('межклассовая матрица разброса: %sx%s ' % (S_B.shape[0], S_B.shape[1]))

ei_vl, ei_ve = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
ei_pa = [(np.abs(ei_vl[i]), ei_ve[:, i])
         for i in range(len(ei_vl))]
ei_pa = sorted(ei_pa, key=lambda k: k[0], reverse=True)
for ei_vl in ei_pa:
    print(ei_vl[0])

tot = sum(ei_vl)
discr = [(i / tot) for i in sorted(ei_vl, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, alpha=0.5, align='center', label='индивидумальная "дискримтмтылвдпы"')
plt.step(range(1, 14), cum_discr, where='mid', label='куммулятивная "дискриминабельность"')
plt.ylabel('доля дискриманбельности')
plt.xlabel('Линейные дискриминанты')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.show()
