import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt

# Зчитуємо зображення
img = cv2.imread('coins_2.JPG')

# Проводимо фільтрацію зображення
filtro = cv2.pyrMeanShiftFiltering(img, 20, 40)
gray = cv2.cvtColor(filtro, cv2.COLOR_BGR2GRAY)

# Виконуємо порогову обробку
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Знаходимо контури
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Знаходимо та видаляємо дрібні області (бурени)
small_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area < 1000:
        small_contours.append(contour)
cv2.drawContours(thresh, small_contours, -1, 0, -1)

# Визначаємо відстані
dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

# Знаходимо локальні максимуми на відстані
local_max = peak_local_max(dist, min_distance=20)

# Перетворимо local_max у маркери
markers = np.zeros_like(thresh, dtype=int)
for idx, point in enumerate(local_max):
    markers[point[0], point[1]] = idx + 1

# Виконуємо водорозподільчу сегментацію
labels = watershed(-dist, markers, mask=thresh)

# Визначення кольорів для маркерів
unique_labels = np.unique(labels)
label_colors = {}
for label in unique_labels:
    if label == 0:
        continue
    color = np.random.randint(0, 256, size=(1, 3))[0]
    label_colors[label] = color

# Створення кольорового зображення маркерів на основі словника
colored_labels = np.zeros_like(img)
for label, color in label_colors.items():
    mask = labels == label
    colored_labels[mask] = color

# Відображення результатів
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ax0, ax1 = axes.ravel()

ax0.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap="gray")
ax0.set_title("Оригінальне зображення")

ax1.imshow(colored_labels)
ax1.set_title("Сегментоване зображення")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
