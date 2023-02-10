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

# One step ahead predictions
results = model.fit()
forecast = results.get_prediction(start=-25)
mean_forecast = forecast.predicted_mean
confidence_intervals = forecast.conf_int()
plt.figure()
plt.plot(dates, mean_forecast.values, color='red', label='forecast')
plt.fill_between(dates, lower_limits, upper_limits, color='pink')
plt.show()

# Dynamic predictions
# uses predicted value, to predict the next value and so on
results.model.fit()
forecast = results.get_prediction(start=-25, dynamic=True)
mean_forecast = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Forecasting out of sample
forecast = results.get_forecast(steps=20)
mean_forecast = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Exercises:
# Generate predictions
one_step_forecast = results.get_prediction(start=-30)

# Extract prediction mean
mean_forecast = one_step_forecast.predicted_mean

# Get confidence intervals of  predictions
confidence_intervals = one_step_forecast.conf_int()

# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:,'lower close']
upper_limits = confidence_intervals.loc[:,'upper close']

# Print best estimate  predictions
print(mean_forecast)

# plot the amazon data
plt.plot(amazon.index, amazon, label='observed')

# plot your mean predictions
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')

# shade the area between your confidence limits
plt.fill_between(lower_limits.index, lower_limits,
		 upper_limits, color='pink')

# set labels, legends and show plot
plt.xlabel('Date')
plt.ylabel('Amazon Stock Price - Close USD')
plt.legend()
plt.show()

# Generate predictions
dynamic_forecast = results.get_prediction(start=-30, dynamic=True)

# Extract prediction mean
mean_forecast = dynamic_forecast.predicted_mean

# Get confidence intervals of predictions
confidence_intervals = dynamic_forecast.conf_int()

# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:,'lower close']
upper_limits = confidence_intervals.loc[:,'upper close']

# Print best estimate predictions
print(mean_forecast)

# plot the amazon data
plt.plot(amazon.index, amazon, label='observed')

# plot your mean forecast
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')

# shade the area between your confidence limits
plt.fill_between(lower_limits.index, lower_limits, 
         upper_limits, color='pink')

# set labels, legends and show plot
plt.xlabel('Date')
plt.ylabel('Amazon Stock Price - Close USD')
plt.legend()
plt.show()

# ARIMA models
# cannot apply an ARMA model to non-stationary time series
# you need to take the difference of the time series to make it stationary
# but then the forecast is trained on the difference and will predict the value
# of the difference of the time series, we want the actual values
# can transform the predictions of the differences (using cumulative sum, integral)
diff_forecast = results.get_forecast(steps=10).predicted_mean
from numpy import cumsum
mean_forecast = cumsum(diff_forecast) + df.iloc[-1,0]

# all of this can be done with ARIMA instead
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df, order=(p,d,q))
# p = number of autoregressive lags
# d = order of dereferencing
# q = moving average order
model.fit()
mean_forecast = results.get_forecast(steps=10).predicted_mean

# Augmented Dickie Fuller test helps us determine how much difference order we need
adf = adfuller(df.iloc[:,0])
print('ADF Statistic:', adf[0])
print('p-value:', adf[1])

adf = adfuller(df.diff().dropna().iloc[:,0])
print('ADF Statistic:', adf[0])
print('p-value:', adf[1])

# Exercises:
# Take the first difference of the data
amazon_diff = amazon.diff().dropna()

# Create ARMA(2,2) model
arma = ARIMA(amazon_diff, order=(2,0,2))

# Fit model
arma_results = arma.fit()

# Print fit summary
print(arma_results.summary())

# Make arma forecast of next 10 differences
arma_diff_forecast = arma_results.get_forecast(steps=10).predicted_mean

# Integrate the difference forecast
arma_int_forecast = np.cumsum(arma_diff_forecast)

# Make absolute value forecast
arma_value_forecast = arma_int_forecast + amazon.iloc[-1,0]

# Print forecast
print(arma_value_forecast)

# Create ARIMA(2,1,2) model
arima = ARIMA(amazon, order=(2,1,2))

# Fit ARIMA model
arima_results = arima.fit()

# Make ARIMA forecast of next 10 values
arima_value_forecast = arima_results.get_forecast(steps=10).predicted_mean

# Print forecast
print(arima_value_forecast)

# ACF & PACF
# Model order is very important for the quality of the forecasts
# ACF - autocorrelation function - lag-1 is the correlation of the time series
    # and the same time series offset by one step
# PACF - partial autocorrelation between a time series and lagged version of itself
    # AFTER we subtract the ffect of correlation at smaller lags
    # correlation associated with just that particular lag, not the others added to it
# if amplitude of the ACF tails off with increasing lag and PACF cuts off after lag p,
    # then we have an AR(p) model
# if the amplitude of the ACF cuts off after lag q and the amplitude of the PACF tails off 
    # then we have an MA(q) model
# if both tail off then we have an ARMA model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8.8))
plot_acf(df, lags=10, zero=False, ax=ax1)
plot_pacf(df, lags=10, zero=False, ax=ax2)
plt.show()
# if the acf values are high and tail off very slowly, this is a sign that the data is non-stationary
    # so it needs to be differenced.
# if the acf at lag-1 is very negative this is a sign we have taken the difference too many times

# Exercises:
# Import
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of df
plot_acf(df, lags=10, zero=False, ax=ax1)

# Plot the PACF of df
plot_pacf(df, lags=10, zero=False, ax=ax2)

plt.show()

# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))

# Plot ACF and PACF
plot_acf(earthquake, lags=10, zero=False, ax=ax1)
plot_pacf(earthquake, lags=10, zero=False, ax=ax2)

# Show plot
plt.show()

# Instantiate model
model = ARIMA(earthquake, order =(1,0,0))

# Train model
results = model.fit()

# AIC & BIC
# Akaike Information Criterion - metric which tells us how good a model is
# Lower AIC indicates a better model
# Penalizes models with lots of parameters (simpler models are better)
# helps prevent over fitting to the training data
# better for predictive models

# Bayesian Information Criterion
# Lower BIC indicates a better model
# Penalizes models with lots of parameters (simpler models are better) more than AIC
# better for explanatory models

# both are printed in the summary table

model = ARIMA(df, ordered=(1,0,1))
results = model.fit()
print(results.summary())

# can also print them out directly
print('AIC: ', results.aic)
print('BIC: ', results.bic)

# we can write loops to fir multiple ARIMA models to a dataset, to find the best model order
for p in range(3):
    for q in range(3):
        model = ARIMA(df, order=(p,0,q))
        results = model.fit()
        print(p,q, results.aic, results.bic)
        # we can add them to a list and convert it into a DF later
        order_aic_bic.append((p,q, results.aic, results.bic))

order_df = pd.DataFrame(order_aic_bic, columns=['p', 'q', 'aic', 'bic'])
# then we can sort my aic values
print(order_df.sort_values('aic'))

# Non stationary model orders & value errors
# leads to a value error  - would result in a non-stationary set of AR coefficients
# we want to skip over these errors, so we should use a try and except block
for p in range(3):
    for q in range(3):
        try:
            model = ARIMA(df, order=(p,0,q))
            results = model.fit()
            print(p,q, results.aic, results.bic)
        except:
            print(p,q, None, None)

# Exercises       
# Create empty list to store search results
order_aic_bic=[]

# Loop over p values from 0-2
for p in range(3):
  # Loop over q values from 0-2
    for q in range(3):
      	# create and fit ARMA(p,q) model
        model = ARIMA(df, order=(p,0,q))
        results = model.fit()
        
        # Append order and results tuple
        order_aic_bic.append((p,q, results.aic, results.bic))

# Construct DataFrame from order_aic_bic
order_df = pd.DataFrame(order_aic_bic, 
                        columns=['p', 'q', 'AIC', 'BIC'])

# Print order_df in order of increasing AIC
print(order_df.sort_values('AIC'))

# Print order_df in order of increasing BIC
print(order_df.sort_values('BIC'))

# Loop over p values from 0-2
for p in range(3):
    # Loop over q values from 0-2
    for q in range(3):
      
        try:
            # create and fit ARMA(p,q) model
            model = ARIMA(earthquake, order=(p,0,q))
            results = model.fit()
            
            # Print order and results
            print(p, q, results.aic, results.bic)
            
        except:
            print(p, q, None, None)     

# Model Diagnostics
# focus on the residuals to the training data
#  difference between our model's one step ahead predictions and the real values of the time series

model = ARIMA(df, order=(p,d,q))
results = model.fit()
residuals = results.resid 
# stored as a pandas series
# Mean absolute error
# how far our the predictions from the real values?
mae = np.mean(np.abs(residuals))
# an ideal model should be uncorreleated with white Gaussian noise centered on zero
# creates the four diagnostic plots
results.plot_diagnostics()
plt.show()
# Residuals plot - if model is working correctly, ther should be no obvious structure
# Histogram plus estimated density plot (shows distribution of the residuals)
    # orange line is smooth version of histogram, green line shows normal distribution.
    # if the model is good, the orange and green line should be almost the same
# Normal QQ plot
    # another way to see the comparision between distribution of model residuals to normal distribution
    # most values should lie along the line
# Correlogram
    # ACF plot of the residuals rather than the data
    # if there is significant correlation in the residuals, it means that there is information
    # in the data that the model hasn't captured
# There are also test statistics in the results.summary() tables
# Prob(Q) - p value for null hypothesis that residuals are uncorrelated
# Prob(JB) - pvalue for null hypothesis that residuals are normally distributed
# if either p value is less than 0.05, we reject the null hypothesis

# Exercises

# Fit model
model = ARIMA(earthquake, order=(1,0,1))
results = model.fit()

# Calculate the mean absolute error from residuals
mae = np.mean(np.abs(results.resid))

# Print mean absolute error
print(mae)

# Make plot of time series for comparison
earthquake.plot()
plt.show()

# Create and fit model
model1 = ARIMA(df, order=(3,0,1))
results1 = model1.fit()

# Print summary
print(results1.summary())

# Box-Jenkins method
# To go from raw data to production model you need:
    # identification, estimation, and model diagnostics
    # Identification -  explore and characterize data to find which is appropriate for ARIMA
        # is it stationary? what transfroms (differencing, log, etc.) will make it stationary?
        # which orders of p,q, are the most promising? (plotting, adfuller())
        # df.plot()
        # adfuller()
        # transforms (df.diff(), np.log(), np.sqrt())
        # plot_acf(), plot_pacf()
    # Estimation - use the data to train the model coefficients
        # model.fit()
        # choose between results.aic, restults.bic
    # Model diagnostics
        # are the residuals uncorrelated
        # are they normally distributed
        # results.plot_diagnostics()
        # results.summary()
    # Decision - 
        # is the model good enough or do we need to go back and rework it?
    # Production - ready to make forecasts
        # results.get_forecast()

# Exercises:

# Plot time series
savings.plot()
plt.show()

# Run Dicky-Fuller test
result = adfuller(savings['savings'])

# Print test statistic
print(result[0])

# Print p-value
print(result[1])

# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of savings on ax1
plot_acf(savings, lags=10, zero=False, ax=ax1)

# Plot the PACF of savings on ax2
plot_pacf(savings, lags=10, zero=False, ax=ax2)

plt.show()

# Loop over p values from 0-3
for p in range(4):
  
  # Loop over q values from 0-3
    for q in range(4):
      try:
        # Create and fit ARMA(p,q) model
        model = ARIMA(savings, order=(p,0,q))
        results = model.fit()
        
        # Print p, q, AIC, BIC
        print(p,q, results.aic, results.bic)
        
      except:
        print(p, q, None, None)

# Create and fit model
model = ARIMA(savings, order=(1,0,2))
results = model.fit()

# Create the 4 diagostics plots
results.plot_diagnostics()
plt.show()

# Print summary
print(results.summary())

# Seasonal Time Series
