from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


DATA_FILE = Path.home() / "Downloads" / "breast+cancer+coimbra" / "dataR2.xlsx"
FEATURE_COLUMNS = ["Age", "BMI"]
TARGET_COLUMN = "Classification"
CLASS_NAMES = {1: "Healthy", 2: "Patient"}
CLASS_COLORS = {1: "blue", 2: "red"}


if not DATA_FILE.exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_FILE}")

df = pd.read_excel(DATA_FILE)

missing_columns = [column for column in [*FEATURE_COLUMNS, TARGET_COLUMN] if column not in df.columns]
if missing_columns:
    raise KeyError(f"Missing columns in dataset: {missing_columns}. Available columns: {list(df.columns)}")

X = df[FEATURE_COLUMNS].values
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1, stratify=y
)

print(df[[*FEATURE_COLUMNS, TARGET_COLUMN]].head())
print()
print(X.shape, y.shape)
print()
print(X_train.shape, y_train.shape)
print()
print(X_test.shape, y_test.shape)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=[CLASS_NAMES[key] for key in sorted(CLASS_NAMES)]))
print(confusion_matrix(y_test, y_pred))
print(
    "Number of mislabeled points out of a total %d points : %d"
    % (X_test.shape[0], (y_test != y_pred).sum())
)

dat = {"y_Actual": y_test, "y_Predicted": y_pred}
dff = pd.DataFrame(dat, columns=["y_Actual", "y_Predicted"])
cross_table = pd.crosstab(
    dff["y_Actual"],
    dff["y_Predicted"],
    rownames=["Actual"],
    colnames=["Predicted"],
    margins=True,
)
print(cross_table)

cross_table_display = cross_table.rename(
    index={key: value for key, value in CLASS_NAMES.items()},
    columns={key: value for key, value in CLASS_NAMES.items()},
)

new_points = np.array([[45, 22.5], [54, 31.2], [62, 27.1], [71, 29.4]])
new_predictions = knn_model.predict(new_points)
print("Predictions for new points:", new_predictions)

point_colors = [CLASS_COLORS.get(label, "gray") for label in y_pred]

plt.figure(1)
plt.scatter(X_test[:, 0], X_test[:, 1], color=point_colors, linewidths=0.1)
plt.xlabel(FEATURE_COLUMNS[0])
plt.ylabel(FEATURE_COLUMNS[1])
plt.title("kNN classification on test data")

plt.figure(2)
plt.scatter(X_test[:, 0], X_test[:, 1], color=point_colors, linewidths=0.1)
for index, point in enumerate(new_points, start=1):
    predicted_class = int(new_predictions[index - 1])
    plt.scatter(
        point[0],
        point[1],
        color="green",
        edgecolors=CLASS_COLORS.get(predicted_class, "black"),
        label=f"Point {index}: {CLASS_NAMES.get(predicted_class, predicted_class)}",
        linewidths=2,
    )
plt.xlabel(FEATURE_COLUMNS[0])
plt.ylabel(FEATURE_COLUMNS[1])
plt.title("New points prediction")
plt.legend()

plt.figure(3, figsize=(7, 3))
plt.axis("off")
plt.title("Cross-validation table", pad=12)
table = plt.table(
    cellText=cross_table_display.values,
    rowLabels=cross_table_display.index,
    colLabels=cross_table_display.columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 1.5)

plt.show()
