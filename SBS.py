from sklearn.base import clone
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

knn = KNeighborsClassifier(n_neighbors=2)


class SBS():
    def __init__(self, estimator, k_f, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_f = k_f
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size,
                                                            random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices = tuple(range(dim))
        self.subsets = [self.indices]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices)
        self.scores_ = [score]

        while dim > self.k_f:
            scores = []
            subsets = []
            for p in combinations(self.indices, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)

                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices = subsets[best]
            self.subsets.append(self.indices)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X[:, self.indices]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df_wine = pd.read_csv(url, header=None)
df_wine.columns = ['Метка класса', 'Алкоголь',
                   'Яблочная кислота', 'Зола',
                   'Щелочность золы', 'Магний',
                   'Всего фенола', 'Флаваноиды',
                   'Фенолы  нефлаводноидные', 'Проантоцианины',
                   'Интенсивность цвета', 'Оттенок'
                                          'OD280/OD315 разбавленных вин', 'Пролин', 'Классность']

from sklearn.model_selection import train_test_split

mms = MinMaxScaler()
stdsc = StandardScaler()

x1, y1 = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=0)

X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

sbs = SBS(knn, k_f=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Верность')
plt.xlabel('Число признаков')
plt.grid()
plt.show()
