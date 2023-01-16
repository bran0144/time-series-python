# ARIMA models and Forecasting
import pandas as pd
import matplotlib as plt

df = pd.read_csv('time_series.csv', index_col='date', parse_dates=True)
fig, ax = plt.subplots()
df.plot(ax=ax)
plt.show()

# Important things to look for: trends, seasonality (fixed cycles), cyclicality (no fixed period)
# White noise - uncorrelated values (does not depend on outcomes that came before)
# Stationarity - distribution of the data doesn't change with time
    # Must meet three criteria:
        # trend stationary - trend is zero (i.e. not trending up or down)
        # variance is constant
        # autocorrelation is constant
# Similar to train/test split in ML, but it's the past data for training and future predictions for test
# Will need to split hte data in time

df_train = df.loc[:'2018']
df_test = df.loc['2019':]

# Exercises:

# Import modules
import pandas as pd
import matplotlib.pyplot as plt

# Load in the time series
candy = pd.read_csv('candy_production.csv', 
            index_col='date',
            parse_dates=True)

# Plot and show the time series on axis ax1
fig, ax1 = plt.subplots()
candy.plot(ax=ax1)
plt.show()

# Split the data into a train and test set
candy_train = candy.loc[:'2006']
candy_test = candy.loc['2007':]

# Create an axis
fig, ax = plt.subplots()

# Plot the train and test sets on the axis ax
candy_train.plot(ax=ax)
candy_test.plot(ax=ax)
plt.show()


# to test for trend stationarity
# augmented Dicky-Fuller test
# tests for trend non-stationarity
# null hypothesis is time series is non-stationary due to trend

from statsmodels.tsa.statstools import adfuller
results = adfuller(df['close'])
print(results)

# 0th element is the test statistics - more negative means more likely to be stationary
# 1st element is the p value - if p value is small (usually smaller than 0.05), reject  null hypothesis
# also includes a dictionary of p values

df_stationary = df.diff().dropna()

# Other transformations may be necessary:
# log
np.log(df)
# square root
np.sqrt(df)
# proportional change
df.shift(1)/df
# try to choose simplest solution whenever possible

# Exercises

# Import augmented dicky-fuller test function
from statsmodels.tsa.stattools import adfuller

# Run test
result = adfuller(earthquake['earthquakes_per_year'])

# Print test statistic
print(result[0])

# Print p-value
print(result[1])

# Print critical values
print(result[4]) 

# Run the ADF test on the time series
result = adfuller(city['city_population'])

# Plot the time series
fig, ax = plt.subplots()
city.plot(ax=ax)
plt.show()

# Print the test statistic and the p-value
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Calculate the first difference of the time series
city_stationary = city.diff().dropna()

# Calculate the second difference of the time series
city_stationary = city.diff().diff().dropna()

# Calculate the first difference and drop the nans
amazon_diff = amazon.diff()
amazon_diff = amazon_diff.dropna()

# Run test and print
result_diff = adfuller(amazon_diff['close'])
print(result_diff)

# Calculate log-return and drop nans
amazon_log = np.log(amazon)
amazon_log = amazon_log.dropna()

# Run test and print
result_log = adfuller(amazon_log['close'])
print(result_log)

# AR, MA, ARMA models
# AR = autoregressive model
# AR(1): yt = a1yt-1 + Et
# order of the model is the number of time lags used
# lag coefficient for AR is used to multiply yt-1
# AR(2):  yt = a1yt-1 + a2yt-2 + Et
# a1 = autoregressive coefficient (in AR1 is just the slope of hte line)

# MA = moving average - regresses the values of time series agsint the previous shock values of same time series
# MA(1): yt = m1Et-1 + Et
# lag coefficient for mA is sued to multiple Et-1
# Et = shock term for current time step
# MA(2): yt = m1Et-1 + m2Et-2 + Et

#  ARMA - autoregressive moving average model
# ARMA = AR + MA
# ARMA(1,1): yt = a1yt-1 + m1Et-1 + Et
# ARMA(p,q) - p is the order of AR part, Q is order of MA

from statsmodels.tsa.arima_process import arma_generate_sample
ar_coefs = [1, -0.5]    #1 = zero lag term 
ma_coefs = [1, 0.2]
y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, scale=0.5)

from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(y, order=(1,0,1))
results = model.fit()

# Import data generation function and set random seed
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(1)

# Set coefficients
ar_coefs = [1]
ma_coefs = [1, -0.7]

# Generate data
y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, scale=0.5)

plt.plot(y)
plt.ylabel(r'$y_t$')
plt.xlabel(r'$t$')
plt.show()

# Remember that the first value of each of the ar_coefs and ma_coefs lists should be 1 for the lag-0 coefficient.
# Remember that an MA(1) model is just an ARMA(0,1) model. Therefore ma_coefs should have a lag-0 and a lag-1 coefficient and ar_coefs should only have a lag-0 coefficient and nothing else (e.g. ma_coefs = [1, ____] and ar_coefs = [1]).

# Import data generation function and set random seed
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(2)

# Set coefficients
ar_coefs = [1, -0.3, -0.2]
ma_coefs = [1]

# Generate data
y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, scale=0.5)

plt.plot(y)
plt.ylabel(r'$y_t$')
plt.xlabel(r'$t$')
plt.show()

# Remember that for lags greater than zero, you need to pass the negative of the desired AR coefficient into the arma_generate_sample() function.

# Import data generation function and set random seed
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(3)

# Set coefficients
ar_coefs = [1, 0.2]
ma_coefs = [1, 0.3, 0.4]

# Generate data
y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, scale=0.5)

plt.plot(y)
plt.ylabel(r'$y_t$')
plt.xlabel(r'$t$')
plt.show()

# Import the ARIMA model
from statsmodels.tsa.arima.model import ARIMA

# Instantiate the model
model = ARIMA(y, order=(1,0,1))

# Fit the model
results = model.fit()

# Fitting time series models
from statsmodels.tsa.arima.model import ARIMA

# ARMA(p,q)  p = autoregressive lags, q moving average lags
model = ARIMA(y, order=(p,0,q))
# if middle term is 0, that's just an ARMA model
# data can be a DF, a pandas series, or numpy array
results = model.fit()
print(results.summary())

# ARMAX - exogenous ARMA
# uses external independent varialbes as well as time series
# like a combo of ARMA model and normal linear regression

# ARMA(1,1)  yt = a1yt-1 + m1Et-1 + Et
# ARMAX(1,1) yt = x1zt + a1yt-1 + m1Et-1 + Et

# Fitting ARMAX
model = ARIMA(df['productivity'], order =(2,0,1), exog=df['hours_sleep'])
results = model.fit()

# Exercises

# Instantiate the model
model = ARIMA(sample['timeseries_1'], order=(2,0,0))

# Fit the model
results = model.fit()

# Print summary
print(results.summary())

# Instantiate the model
model = ARIMA(earthquake, order=(3,0,1))

# Fit the model
results = model.fit()

# Print model fit summary
print(results.summary())

# Instantiate the model
model = ARIMA(hospital['wait_times_hrs'], order=(2,0,1), exog=hospital['nurse_count'])

# Fit the model
results = model.fit()

# Print model fit summary
print(results.summary())

