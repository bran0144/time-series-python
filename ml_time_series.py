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

model = LinearSVC()
model.fit(X,y)
# it is common for y to be of shape (samples, 1)
# X is training data, y (labels for each datapoint)

model.coef_ 
# returns coefficients for each feature
# once your model is fit, you can call the .predict method

predictions = model.predict(X_test)

# Exercises
from sklearn.svm import LinearSVC

# Construct data for the model
X = data[['petal length (cm)', 'petal width (cm)']]
y = data[['target']]

# Fit the model
model = LinearSVC()
model.fit(X, y)

# Create input array
X_predict = targets[['petal length (cm)', 'petal width (cm)']]

# Predict with the model
predictions = model.predict(X_predict)
print(predictions)

# Visualize predictions and actual values
plt.scatter(X_predict['petal length (cm)'], X_predict['petal width (cm)'],
            c=predictions, cmap=plt.cm.coolwarm)
plt.title("Predicted class values")
plt.show()

from sklearn import linear_model

# Prepare input and output DataFrames
X = housing[['MedHouseVal']]
y = housing[['AveRooms']]

# Fit the model
model = linear_model.LinearRegression()
model.fit(X,y)

from sklearn import linear_model

# Prepare input and output DataFrames
X = housing[['MedHouseVal']]
y = housing[['AveRooms']]

# Fit the model
model = linear_model.LinearRegression()
model.fit(X,y)

# combining time series data with machine learning
# get to know your data
# audio data

from glob import glob
files = glob('data/heartbeat-sounds/files/*.wav')

import librosa as lr 
audio, sfreq = lr.load('data/heartbeat-sounds/proc/files/murmur__201101051104.wav')

# Create a time array (option 1)
indices = np.arrange(0, len(audio))
time = indices/sfreq

# Create a time array (option 2)
final_time = (len(audio) - 1) /sfreq
time = np.linspace(0, final_time, sfreq)

# exploring stock data
data = pd.read_csv('path/to/data.csv')
data.columns
data.head()
df['date'].dtypes

df['date'] = pd.to_datetime(df['date'])

# Exercises
import librosa as lr
from glob import glob

# List all the wav files in the folder
audio_files = glob(data_dir + '/*.wav')

# Read in the first audio file, create the time array
audio, sfreq = lr.load(audio_files[0])
time = np.arange(0, len(audio)) / sfreq

# Plot audio over time
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()

# Read in the data
data = pd.read_csv('prices.csv', index_col=0)

# Convert the index of the DataFrame to datetime
data.index = pd.to_datetime(data.index)
print(data.head())

# Loop through each column, plot its values over time
fig, ax = plt.subplots()
for column in data:
    data[column].plot(ax=ax, label=column)
ax.legend()
plt.show()

# Classifying and feature engineering

