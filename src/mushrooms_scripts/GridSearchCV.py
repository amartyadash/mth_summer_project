import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../datasets/mushrooms.csv')
labelencoder = LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

X = data.drop(columns=['class'])
y = list(data['class'])
knn = KNeighborsClassifier()
param_grid = {'n_neighbors':np.arange(1,25)}
knn_GSCV = GridSearchCV(knn, param_grid, cv=5)
knn_GSCV.fit(X, y)
print(knn_GSCV.best_params_)
print('Accuracy % = {}'.format((knn_GSCV.best_score_)*100))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
print('Prediction: {}'.format(list(knn_GSCV.predict(X_test))))
print('Actual:{}'.format(y_test))

# print(data.head())
# print(data['stalk-color-above-ring'].unique())