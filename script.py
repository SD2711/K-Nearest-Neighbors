import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Загружаем данные
df = pd.read_excel("C:/Users/user/Downloads/breast+cancer+coimbra/dataR2.xlsx")

# Выбираем два признака для визуализации и классификации
X = df[["BMI", "Glucose"]]
y = df["Classification"]

# Формируем обучающую и тестовую выборки
# stratify=y сохраняет доли классов в обеих выборках
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1, stratify=y
)

print(df[["BMI", "Glucose", "Classification"]].head())
print()
print(X.shape, y.shape)
print()
print(X_train.shape, y_train.shape)
print()
print(X_test.shape, y_test.shape)

# Применяем метод kNN
# Так как метод основан на расстояниях, сначала стандартизируем признаки
KNN_model = Pipeline(
    [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=11))]
)

KNN_model.fit(X_train, y_train)
y_pred = KNN_model.predict(X_test)

print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred, labels=[1, 2]))
print()
print("Точность модели:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Количество ошибок:", (y_test != y_pred).sum())

# Формируем кросс-валидационную таблицу
dat = {"y_Actual": y_test.to_numpy(), "y_Predicted": y_pred}
dff = pd.DataFrame(dat, columns=["y_Actual", "y_Predicted"])

cross_table = pd.crosstab(
    dff["y_Actual"],
    dff["y_Predicted"],
    rownames=["Actual"],
    colnames=["Predicted"],
    margins=True,
)
print()
print(cross_table)

# Задаём новые точки для прогнозирования
n = np.array([[21.0, 70.0], [24.0, 82.0], [30.0, 120.0], [35.0, 160.0]])

# Делаем прогноз для новых точек
new_points = KNN_model.predict(n)
print()
print("Прогноз для новых точек:")
print(new_points)

# Визуализируем исходные данные
plt.figure(figsize=(8, 5))
colors = ["blue" if cls == 1 else "red" for cls in y]
plt.scatter(X["BMI"], X["Glucose"], color=colors, linewidths=0.3)
plt.xlabel("BMI, кг/м²")
plt.ylabel("Glucose, мг/дл")
plt.title("Облако точек по двум признакам")
plt.show()

# Визуализируем тестовую выборку
plt.figure(figsize=(8, 5))
test_colors = ["blue" if cls == 1 else "red" for cls in y_pred]
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], color=test_colors, linewidths=0.3)
plt.xlabel("BMI, кг/м²")
plt.ylabel("Glucose, мг/дл")
plt.title("Результат классификации тестовой выборки")
plt.show()

# Визуализируем новые точки
plt.figure(figsize=(8, 5))
plt.scatter(X["BMI"], X["Glucose"], color=colors, linewidths=0.2, alpha=0.4)

plt.scatter(n[0, 0], n[0, 1], color="green", label="Точка 1", s=90)
plt.scatter(n[1, 0], n[1, 1], color="purple", label="Точка 2", s=90)
plt.scatter(n[2, 0], n[2, 1], color="orange", label="Точка 3", s=90)
plt.scatter(n[3, 0], n[3, 1], color="brown", label="Точка 4", s=90)

plt.xlabel("BMI, кг/м²")
plt.ylabel("Glucose, мг/дл")
plt.title("Новые объекты на плоскости признаков")
plt.legend()
plt.show()
