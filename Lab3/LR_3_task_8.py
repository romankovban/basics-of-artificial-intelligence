from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Завантаження даних (у цьому випадку - даних про квіти ірису)
iris = load_iris()
X = iris.data  # Ознаки (вектори ознак)
y = iris.target  # Мітки класів

# Ініціалізація моделі KMeans з 5 кластерами
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=None)

# Навчання моделі на вхідних даних
kmeans.fit(X)

# Прогноз кластерів для кожного зразка
y_kmeans = kmeans.predict(X)

# Візуалізація результатів кластеризації
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


# Визначення функції для пошуку кластерів без використання бібліотеки KMeans
def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        labels = pairwise_distances_argmin(X, centers)
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels


# Визначення центрів кластерів та міток для них
centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

# Знову визначення центрів кластерів та міток для них з іншим random_state
centers, labels = find_clusters(X, 3, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

# Використання KMeans з бібліотеки scikit-learn для кластеризації
labels = KMeans(3, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

# Відображення графіка
plt.show()
