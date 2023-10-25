import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

# Вхідний файл, який містить дані
input_file = 'data_multivar_regr.txt'

# Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбивка даних на навчальний та тестовий набори (80% / 20%)
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Тренувальні дані
X_train, y_train = X[:num_training], y[:num_training]

# Тестові дані
X_test, y_test = X[num_training:], y[num_training:]

# Створення об'єкта лінійного регресора та навчання на тренувальних даних
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Прогнозування результату для тестового набору даних
y_test_pred = regressor.predict(X_test)

# Виведення на екран метрик якості лінійної регресії
print("Linear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Створення поліноміального регресора ступеня 10 та навчання його на тренувальних даних
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)

poly_regressor = linear_model.LinearRegression()
poly_regressor.fit(X_train_transformed, y_train)

# Перетворення вибіркової точки даних та прогноз для неї за допомогою поліноміального регресора
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.fit_transform(datapoint)

# Прогноз за допомогою лінійного та поліноміального регресорів
linear_prediction = regressor.predict(datapoint)
poly_prediction = poly_regressor.predict(poly_datapoint)

# Виведення прогнозів
print("\nLinear regression prediction:", linear_prediction)
print("Polynomial regression prediction:", poly_prediction)
