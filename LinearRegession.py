# from scratch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from library
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error


plt.rcParams['figure.figsize'] = (20.0, 10.0)
data = pd.read_csv('Sample4.csv')
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

print(b1,b0)