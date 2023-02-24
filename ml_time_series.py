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
# always visualize raw data before fitting models

ixs = np.arange(audio.shape[-1])
time = ixs / sfreq
fig, ax = plt.subplots()
ax.plot(time, audio)

# raw timeseries dat is too noisy for classification
# we need to calculat features
# where to start? summarize your audio data
# use min, max, avg of each sample
# axis =-1 collapses across the last dimension (time)
means = np.mean(audio, axis=-1)
maxs = np.max(audio, axis=-1)
stds = np.std(audio, axis=-1)

# result is an array per timeseries
# if we have a label for each sample, we can use scikit-learn to create and fit a classifier
# prepare for scikit learn
from sklearn.svm import LinearSVC
X = np/column_stack([means, maxs, stds])
y = labels.reshape(-1,1)
model = LinearSVC()
model.fit(X,y)

# score the classifier
from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
percent_score = sum(predictions == labels_test) /len(labels_test)
percent_score = accuracy_score(labels_test, predictions)


# exercises
fig, axs = plt.subplots(3, 2, figsize=(15, 7), sharex=True, sharey=True)

# Calculate the time array
time = np.arange(normal.shape[0]) / sfreq

# Stack the normal/abnormal audio so you can loop and plot
stacked_audio = np.hstack([normal, abnormal]).T

# Loop through each audio file / ax object and plot
# .T.ravel() transposes the array, then unravels it into a 1-D vector for looping
for iaudio, ax in zip(stacked_audio, axs.T.ravel()):
    ax.plot(time, iaudio)
show_plot_and_make_titles()

# Average across the audio files of each DataFrame
mean_normal = np.mean(normal, axis=1)
mean_abnormal = np.mean(abnormal, axis=1)

# Plot each average over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
ax1.plot(time, mean_normal)
ax1.set(title="Normal Data")
ax2.plot(time, mean_abnormal)
ax2.set(title="Abnormal Data")
plt.show()

from sklearn.svm import LinearSVC

# Initialize and fit the model
model = LinearSVC()
model.fit(X_train, y_train)

# Generate predictions and score them manually
predictions = model.predict(X_test)
print(sum(predictions == y_test.squeeze()) / len(y_test))

# Improving featres for classification
# auditory envelope - calculates the amplitude, then smooths is over time
# first, remove noise in timeseries data by smoothing is with a rolling window
# takes the mean of each window
print(audio.shape)
window_size = 50
windowed = audio.rolling(window=window_size)
audio_smooth = windowed.mean()

# calculate the auditory envelope
audio_recitified = audio.apply(np.abs)
audio_envelope = audio_recitified.rolling(50).mean()

# feature engineering the envelope
envelope_mean = np.mean(audio_envelope, axis=0)
envelope_std = np.std(audio_envelope, axis=0)
envelope_max = np.max(audio_envelope, axis=0)

# create training data
X = np.column_stack([envelope_mean, envelope_std, envelope_max])
y = labels.reshape(-1,1)

# calculate cross-val score
# splits data into training/validation sets
# fits model on training data
# scores it on validation data
# repeats the process

from sklearn.model_selection import cross_val_score

model = LinearSVC()
scores = cross_val_score(model, X, y, cv=3)
print(scores)

# tempogram, estimates the tempo of sound over time
import librosa as lr 
audio_tempo = lr.beat.tempo(audio, sr=sfreq, hop_length=2**6, aggregate=None)

# Exercises

# Plot the raw data first
audio.plot(figsize=(10, 5))
plt.show()

# Rectify the audio signal
audio_rectified = audio.apply(np.abs)

# Plot the result
audio_rectified.plot(figsize=(10, 5))
plt.show()

# Smooth by applying a rolling mean
audio_rectified_smooth = audio_rectified.rolling(50).mean()

# Plot the result
audio_rectified_smooth.plot(figsize=(10, 5))
plt.show()

# Calculate stats
means = np.mean(audio_rectified_smooth, axis=0)
stds = np.std(audio_rectified_smooth, axis=0)
maxs = np.max(audio_rectified_smooth, axis=0)

# Create the X and y arrays
X = np.column_stack([means, stds, maxs])
y = labels.reshape(-1, 1)

# Fit the model and score on testing data
from sklearn.model_selection import cross_val_score
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))

# Calculate the tempo of the sounds
tempos = []
for col, i_audio in audio.items():
    tempos.append(lr.beat.tempo(i_audio.values, sr=sfreq, hop_length=2**6, aggregate=None))

# Convert the list to an array so you can manipulate it more easily
tempos = np.array(tempos)

# Calculate statistics of each tempo
tempos_mean = tempos.mean(axis=-1)
tempos_std = tempos.std(axis=-1)
tempos_max = tempos.max(axis=-1)

# Create the X and y arrays
X = np.column_stack([means, stds, maxs, tempos_mean, tempos_std, tempos_max])
y = labels.reshape(-1, 1)

# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))

# Spectrogram


