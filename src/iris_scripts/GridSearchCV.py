import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('../datasets/iris.csv')
# print(df.head())
X = df.drop(columns=['y'])
# print(X.head())
y = list(df['y'])
# print(y[0:5])
knn = KNeighborsClassifier()
param_grid = {'n_neighbors':np.arange(1,25)}
knn_GSCV = GridSearchCV(knn, param_grid, cv=5)
knn_GSCV.fit(X,y)
print(knn_GSCV.best_params_)
print('Accuracy % = {}'.format((knn_GSCV.best_score_)*100))
# Just to check values we can use train_test_split but use GSCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
print('Prediction: {}'.format(list(knn_GSCV.predict(X_test))))
print('Actual:{}'.format(y_test))