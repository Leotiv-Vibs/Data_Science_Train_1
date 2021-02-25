import pandas as pd
import numpy as np
from io import StringIO

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder

csv_data = '''A,B,C,D
              1.0,2.0,3.0,4.0
              5.0,6.0,,8.0
              10.0,11.0,12.0'''

# df = pd.read_csv(StringIO(csv_data))
# df = df.dropna()
# print(df)
# from sklearn.impute import SimpleImputer
#
# imr = SimpleImputer(missing_values='NaN', strategy='mean', verbose=0)
# imr = imr.fit(df)
# imputed_data = imr.transform(df.values)
# print(imputed_data)
df = pd.DataFrame([['зеленый', 'M', '10.1', 'класс1'],
                   ['красный', 'L', '13.5', 'класс2'],
                   ['синий', 'XL', '15.3', 'класс1']])
df.columns = ['цвет', 'размер', 'цена', 'метка']
size_mapping = \
    {
        'M': 1,
        'XL': 3,
        'L': 2,
    }
df['размер'] = df['размер'].map(size_mapping)
# print(df)
class_mappping = {label: idx for idx, label in enumerate(np.unique(df['метка']))}
df['метка'] = df['метка'].map(class_mappping)
# print(df)
# class_le = LabelEncoder()
# y = class_le.fit_transform(df['метка'].values)
# print(y)
x = df[['цвет', 'размер', 'цена']].values
# color_le = LabelEncoder()
# x[:, 0] = color_le.fit_transform(x[:, 0])
ohe = ColumnTransformer(
    transformers=[
        ("OneHot",  # Just a name
         OneHotEncoder(),  # The transformer class
         [0]  # The column(s) to be applied on.
         )
    ]
)

# print(ohe.fit_transform(x))
#
# print(pd.get_dummies(df[['цена', 'цвет', 'размер']]))

# print(x)
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
# print(X_train_norm, X_test_norm)
#
# print('Метки классов: ', np.unique(df_wine['Метка класса']))
# print(df_wine.head())
from sklearn.linear_model import LogisticRegression

LogisticRegression(penalty='l1')
lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)

print(lr.score(X_train_std, y_train))
print(lr.score(X_test_std, y_test))
print()
