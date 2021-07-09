# 3 Exercises for Supervised Machine Learning

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# 1. Feature Engineering
#1a) reading boston data
boston = load_boston()
X = boston['data']
y = boston['target']

#1b) extracting polynomial features
poly = PolynomialFeatures(2, include_bias=False)
#poly.fit_transform(X)
poly = PolynomialFeatures(interaction_only=True)
poly.fit_transform(X)

#print(X)
#1c) creating DataFrame

df_p = pd.DataFrame(poly.fit_transform(X), columns=poly.get_feature_names(boston["feature_names"]))

df_p['y'] = y
df_p.to_csv('output/polynomials.csv')

#2
#2a) reading data from ex.1
df_p = pd.read_csv('output/polynomials.csv')

#2b) assigning variables to X and y
X = df_p.iloc[:, 0:]
y = df_p['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
#2c) learning different models
# Linear Regression

lm = LinearRegression().fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))
print(lm.coef_)
df_p["lm"] = pd.DataFrame(lm.coef_)

# Ridge Regression

ridge = Ridge(alpha=0.3).fit(X_train, y_train)
print(ridge.coef_)
df_p["ridge"] = pd.DataFrame(ridge.coef_)

# Lasso

lasso = Lasso(alpha=0.3).fit(X_train, y_train)
df_p["lasso"] = pd.DataFrame(lasso.coef_)

#2d) creating dataframe

#df_all = pd.DataFrame(lasso.coef_, ridge.coef_, lm.coef_, index="feature_names", columns=["lasso", "ridge", "lm"])

#2e) creating plot

plt.show()
plt.savefig('output/polynomials.pdf')


#3 Neural Network Regression
#3a) loading dataset
diabetes = load_diabetes()

#print(diabetes.keys())

X = diabetes["data"]
y = diabetes["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#3b) learning a Neural Network Regressor
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#unfortunately pipeline+ grid search did not work but would be the far better solution:
mlp = MLPRegressor(hidden_layer_sizes=(100,100,100), alpha=0.0001, activation='identity', solver='adam', max_iter=1000)
mlp2 = MLPRegressor(hidden_layer_sizes=(10,10,10), alpha=0.05, activation='identity', solver='adam', max_iter=1000)
mlp3 = MLPRegressor(hidden_layer_sizes=(15,15,), alpha=0.0001, activation='identity', solver='adam', max_iter=1000)
mlp4 = MLPRegressor(hidden_layer_sizes=(50,50,50), alpha=0.0001, activation='identity', solver='adam', max_iter=1000)
mlp5 = MLPRegressor(hidden_layer_sizes=(10,10,10), alpha=0.0001, activation='relu', solver='adam', max_iter=1000)
mlp6 = MLPRegressor(hidden_layer_sizes=(10,10,10), alpha=0.0001, activation='identity', solver='lbfgs', max_iter=1000)
mlp7 = MLPRegressor(hidden_layer_sizes=(10,10,10), alpha=0.0001, activation='identity', solver='adam', max_iter=1000)
mlp8 = MLPRegressor(hidden_layer_sizes=(10,10,10), alpha=0.0001, activation='identity', solver='adam', max_iter=1000)
mlp9 = MLPRegressor(hidden_layer_sizes=(10,10,10), alpha=0.0001, activation='identity', solver='sgd', max_iter=1000)

mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))

mlp2.fit(X_train, y_train)
print(mlp2.score(X_test, y_test))

mlp9.fit(X_train, y_train)
print(mlp2.score(X_test, y_test))
#3c)best parameters: The first ones are the best parameters

#3d) plotting heatmap
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(8))
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")

plt.show()
plt.savefig('output/nn_diabetes_importances.pdf')

#4 Neural Networks Classification
# 4a) loading dataset
cancer = load_breast_cancer()
# print(cancer.keys())

X = cancer["data"]
y = cancer["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 4b) reading about ROC-AUC-metric: check

# 4c) learning a a Neural Network Classifier
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
mlp = MLPClassifier(random_state=0, hidden_layer_sizes=(10, 10), alpha=0.001)
#mlp = MLPClassifier(random_state=0, hidden_layer_sizes=(15, 15), alpha=0.0001)
#mlp = MLPClassifier(random_state=0, hidden_layer_sizes=(100, 100), alpha=0.01)
#mlp = MLPClassifier(random_state=0, hidden_layer_sizes=(5, 5), alpha=0.01)

mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))

rac = roc_auc_score(y, mlp.predict_proba(X)[:, 1])
print(rac)
#ROC AuC Score is very close to 1: 0.9763226045135036

preds = mlp.predict(X_test)
confusion_m = confusion_matrix(y_test, preds)
hm = sns.heatmap(confusion_m, annot=True)

#4d) plotting confusion matrix
plt.show()
plt.savefig('output/nn_breast_confusion.pdf')

