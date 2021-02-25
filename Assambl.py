from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importens = forest.feature_importances_
indices = np.argsort(importens)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]],
                            importens[indices[f]]
                            ))
plt.title('Важности признаков')
plt.bar(range(X_train.shape[1]), importens[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

forest.fit(X_train, y_train)
pred = forest.predict(X_test)
print(pred)
