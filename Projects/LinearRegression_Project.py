import numpy as np
import pandas as pd
import seaborn as se
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.datasets import fetch_california_housing


''' Problem statement:
you have been given a dataset that describes the house in Boston. Now base on the given features, you have to predict  
the house price.
# '''

# Creating a data frame
california = fetch_california_housing(as_frame=True)
# print(fetch_california_housing().DESCR)

# Exploratory Data Analysis
df = pd.DataFrame(california.data)
# print(df.head())

# Adding column names
df.columns = fetch_california_housing().feature_names
# print(df.head())
# print(california.frame.head())

# Target value
# print(california.target.head())

# Table overview
# print(df.info())
# print(california.frame.info())
# print(df.nunique())  # unique values
# print(df.isnull().sum())  # checking null values
# print(df.describe())  # statical calculations

# correlation is used to find and understand the relation between variable and features in datasets
# print(df.corr())
# but in a table form we cannot understand the relations between variables and features so...

# Visualization
''' 1. HeatMap
plt.figure(figsize=(10, 10))
se.heatmap(data=df.corr(), annot=True)
# '''

''' 2.Histograms
california.frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)
# '''

''' 3. PairPlot
se.pairplot(df, size=5)
# '''

plt.show()

# Exporting the dataset
# df.to_csv('california_dataset.csv')
