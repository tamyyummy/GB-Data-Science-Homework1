import seaborn as sns
import numpy as np
data = sns.load_dataset('tips')

X = np.array(data['total_bill']).reshape(-1, 1)
y = data['tip'].astype(np.int64)

# Дополните код для работы программы на месте троеточия "..."

# Импортируйте функцию train_test_split и логистическую регрессию
# А также MSE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, roc_auc_score

# Разделите данные на тренировочную и тестовые выборки, с отношением 20%

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = .2)

# Инициализируйте и обучите модель на тренировочной выборке 

model = LogisticRegression(random_state = 42)

model.fit(X_train, y_train)

# Сделайте предсказание на тестовой выборке

pred = model.predict(X_test)

# Выведите среднеквадратичную ошибку предсказания

print('MSE: ', mean_squared_error(pred, y_test))

