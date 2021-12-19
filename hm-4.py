import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

data = sns.load_dataset('tips')

X = data.drop('total_bill', axis = 1)
y = data['tip']

# Дополните код для работы программы на месте троеточия "..."


# Импортируйте энкодеры для предобработки категориальных переменных

from sklearn.preprocessing import OrdinalEncoder

# Импортируйте регрессор на дереве принятия решений

from sklearn.tree import DecisionTreeRegressor

# Импортируйте MAE

from sklearn.metrics import mean_absolute_error

# Предобработайте категориальные переменные в исходных данных (может быть более одной строки)

encoder = OrdinalEncoder()

for column in ['sex', 'smoker', 'day', 'time']:
    X[column] = encoder.fit_transform(X[[column]])

# Разделите данные на тренировочную и тестовые выборки

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = .2)

# Создайте и обучите регрессор на тренировочных данных

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Сделайте предсказание на тренировочных данных

pred = model.predict(X_test)

# Вывести MAE до 3-го знака после запятой

print(f'MAE fold : {mean_absolute_error(pred, y_test):.3f}')
