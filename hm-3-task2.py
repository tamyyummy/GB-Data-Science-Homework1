import seaborn as sns
import numpy as np
data = sns.load_dataset('tips')

X = np.array(data['total_bill']).reshape(-1, 1)
y = data['tip']

# Импортируйте кросс-валидацию

from sklearn.model_selection import cross_validate

from sklearn.linear_model import LinearRegression

# Инициализируйте кросс-валидацию с 5ю фолдами

lin_reg = LinearRegression()

cv = cross_validate(lin_reg, X, y, cv = 5)

# Выведите результаты кросс-валидации и среднее

print(cv['test_score'])

print("Среднее отклонение:", cv['test_score'].mean())