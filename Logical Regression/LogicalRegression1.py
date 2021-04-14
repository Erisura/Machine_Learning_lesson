import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error

data = pd.read_csv('boston.csv')
data_used = data[['crim', 'rm', 'lstat', 'medv']]
y = np.array(data_used['medv'])
features = data_used.drop('medv', axis = 1)
X = np.array(features)

X_df = pd.DataFrame(data, columns=['crim', 'rm', 'lstat', 'medv'])
print(X_df.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train)
print("回归方程系数:{}".format(lr.coef_))
print("回归方程截距:{}".format(lr.intercept_))

y_pred_test = lr.predict(X_test)
y_pred_train = lr.predict(X_train)

test_mae = mean_absolute_error(y_pred_test, y_test)
test_mse = mean_squared_error(y_pred_test, y_test)
print("test mae: " + str(test_mae))
print("test mse: " + str(test_mse))

train_mae = mean_absolute_error(y_pred_train, y_train)
train_mse = mean_squared_error(y_pred_train, y_train)
print("train_mae: " + str(train_mae))
print("train_mse: " + str(train_mse))

plt.scatter(y_test, y_pred_test)
plt.plot([0, 50], [0, 50])  # (0,0)到(50,50)的线 K为黑色
plt.show()