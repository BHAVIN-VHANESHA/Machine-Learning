"""
Linear regression is a Predictive Analysis, it is Supervised Learning Model
y = mx + c [m=slope, c=intercept]

cost function = expected_value - predicted_value
Types:
-> MSE stands for Mean Square Error = 1/n(actual_value - predictive_value)^2
   which is also known as L2 loss
-> MAE stands for Mean Absolute Error = 1/n|actual_value - predictive_value|,
   which is also known as L1 loss
-> RMSE stands for Root Mean Square Error = root(MSE)
-> RSE stands for Relative Square Error = (actual_value - predictive_value)^2 / actual_value - mean(actual_value))^2
-> RAE stands for Relative Absolute Error = |actual_value - predictive_value| / |actual_value - mean(actual_value)|
-> R^2 = 1 - RSE it is a popular metric for the accuracy of the model, it represents how close the data values are to be
   fitted on the regression line, higher the R-squared better the model fits the data

Gradient Descent: it updates the value of m & c to reduce the cost function
"""

# from scratch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from library
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error


var_x = np.array([1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0, 4.0, 4.2, 4.5, 4.4, 4.8, 5.0, 5.5, 5.6,
                  5.8, 5.9, 6.0, 6.1, 6.6, 6.7, 7.0, 7.3, 8.9, 9.1])
var_y = np.array([39343, 46205, 37731, 43535, 39821, 56642, 60150, 54445, 64445, 57189, 63218, 55794, 56957, 42218,
                  54321, 68764, 78747, 63875, 95752, 12554, 76002, 70056, 98871, 96645, 87651, 10245, 14909, 15472,
                  58761, 96475])
# print(len(var_x))
# print(len(var_y))
# plt.scatter(var_x, var_y)
# plt.show()

# df = pd.DataFrame({"Experience": var_x, "Salary": var_y})
# print(df)
# print(df.info())

X = var_x.reshape(-1, 1)
Y = var_y
# print(X)
# print(Y)

# splitting the data into training & testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3)

# choosing the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# training the model
model.fit(X_train, Y_train)

# predicting the model
print(Y_test)  # expected output
Y_pred = model.predict(X_test)

# performance matrix MSE
from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_test, Y_pred))

# accuracy
print(model.score(X_test, Y_test))
from sklearn.metrics import r2_score
print(r2_score(Y_test, Y_pred))


'''
plt.rcParams['figure.figsize'] = (20.0, 10.0)
data = pd.read_csv('Datasets/sample4.csv')
print(data.shape)
data.head()

X = data.values
Y = data.values

mean_x = np.mean(X)
mean_y = np.mean(Y)

n = len(X)

number = 0
denom = 0
for i in range(0):
    number += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2

b1 = number / denom
b0 = mean_y - (b1 * mean_x)

print(b1, b0)
'''
