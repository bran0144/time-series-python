# ML for Time Series Data

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv')
data.head()

fig, ax = plt.subplots(figsize=(12,6))
data.plot('date', 'close', ax=ax)
ax.set(title="AAPL daily closing price")

# period = amound of time that passes between timestamps

# ML pipeline steps
    # feature extraction
    # model fitting
    # prediction and validation

# Exercises

# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(y='data_values', ax=axs[0])
data2.iloc[:1000].plot(y='data_values', ax=axs[1])
plt.show()

# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(x='time', y='data_values', ax=axs[0])
data2.iloc[:1000].plot(x='time', y='data_values', ax=axs[1])
plt.show()

# Machine Learning Basics
# first, you should alaways look at raw data using array.shape or df.head()
# you should also visualize your data to make sure it looks as you'd expect
# using matplotlib
# fig, ax = plt.subplots()
# ax.plot(...)

# using pandas
# fig, as = plt.subplots()
# df.plot(..., ax = ax)

# histograms and scatterplots are good places to start
# look at distribution of your data
# look at outliers or missing data

from sklearn.svm import LinearSVC

# scikit-learn expects a particular structure of data (samples, features)
# data needs to be two dimensional (at least)
# make sure first dimension is samples
# if your data isn't this shape, you can reshape it to use sklearn

# most common is to "transpose" your data
array.T.shape
# -1 will automatically fill that axis with remaining values
array.reshape(-1,1).shape

