import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Завантаження даних про діабет
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Поділ даних на навчальні та тестову вибірки
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

# Створення моделі лінійної регресії та навчання її на навчальних даних
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

# Прогноз тестової вибірки
ypred = regr.predict(Xtest)

# Розрахунок та вивід коефіцієнтів регресії
print("Коефіцієнти регресії:", regr.coef_)
print("Перетин лінії:", regr.intercept_)
print("Коефіцієнт кореляції R^2:", r2_score(ytest, ypred))
print("Середня абсолютна помилка (MAE):", mean_absolute_error(ytest, ypred))
print("Середньоквадратична помилка (MSE):", mean_squared_error(ytest, ypred))

# Побудова графіка для порівняння спостережуваних значень і передбачених значень
fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()
