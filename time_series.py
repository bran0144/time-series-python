# Time Series Analysis in Python

# changing index to datetime
df.index=pd.to_datetime(df.index)

# join 2 DF's 
df1.join(df2)

df['col'].pct_change()
df['col'].diff()

df['ABC'].corr(df['XYZ'])

# Exercises
# Import pandas and plotting modules
import pandas as pd
import matplotlib.pyplot as plt

# Convert the date index to datetime
diet.index = pd.to_datetime(diet.index)

# From previous step
diet.index = pd.to_datetime(diet.index)

# Plot the entire time series diet and show gridlines
diet.plot(grid=True)
plt.show()

# From previous step
diet.index = pd.to_datetime(diet.index)

# Slice the dataset to keep only 2012
diet2012 = diet['2012']

# Plot 2012 data
diet2012.plot(grid=True)
plt.show()

# Import pandas
import pandas as pd

# Convert the stock index and bond index into sets
set_stock_dates = set(stocks.index)
set_bond_dates = set(bonds.index)

# Take the difference between the sets and print
print(set_stock_dates - set_bond_dates)

# Merge stocks and bonds DataFrames using join()
stocks_and_bonds = stocks.join(bonds, how='inner')

# Correlations of two time series
# Be careful with correlations - for stocks, you should look at their returns, not their levels
# this gives the returns instead of prices (use pct_change())
df['SPX_ret'] = df['SPX_Prices'].pct_change()
df['R2000_Ret'] = df['R2000_Prices'].pct_change()

plt.scatter(df['SPX_Ret'], df['R2000_ret'])
plt.show()

correlation = df['SPX_Ret'].corr(df['R2000_Ret'])
print("Correlation is: ", correlation)

# Exercises:

# Compute percent change using pct_change()
returns = stocks_and_bonds.pct_change()

# Compute correlation using corr()
correlation = returns['SP500'].corr(returns['US10Y'])
print("Correlation of stocks and interest rates: ", correlation)

# Make scatter plot
plt.scatter(returns['SP500'], returns['US10Y'])
plt.show()

# Compute correlation of levels
correlation1 = levels['DJI'].corr(levels['UFO'])
print("Correlation of levels: ", correlation1)

# Compute correlation of percent changes
changes = levels.pct_change()
correlation2 = changes['DJI'].corr(changes['UFO'])
print("Correlation of changes: ", correlation2)

# Simple Linear Regression

# OLS - Ordinary Least Squares
# python packages that do regression
import statsmodels.api as sm 
sm.OLS(x,y).fit()

# from numpy
np.polyfit(x,y, deg=1)

# from pandas
pd.ols(y,x)

# from scipy
from scipy import stats 
stats.linregress(x,y)

# beware that the order of x,y is not consistent

import statsmodels.api as sm 
df['SPX_Ret'] = df['SPX_Prices'].pct_change()
df['R2000_Ret'] = df['R2000_Prices'].pct_change()
df = sm.add_constant(df)  #adds a constant for the regression intercept
df = df.dropna() #drops the rows with na

results = sm.OLS(df['R2000_Ret'],df[['const','SPX_Ret']]).fit()
print(results.summary())

# intercept
results.params[0]
# slope
restuls.params[1]

# magnitude of the correlation is the square root of the r quared
# sign of the correlation is the sign of the slope of the regression line

# Exercises
# Import the statsmodels module
import statsmodels.api as sm

# Compute correlation of x and y
correlation = x.corr(y)
print("The correlation between x and y is %4.2f" %(correlation))

# Convert the Series x to a DataFrame and name the column x
dfx = pd.DataFrame(x, columns=['x'])

# Add a constant to the DataFrame dfx
dfx1 = sm.add_constant(dfx)

# Regress y on dfx1
result = sm.OLS(y, dfx1).fit()

# Print out the results and look at the relationship between R-squared and the correlation above
print(result.summary())

# Autocorrelation (aka serial correlation)
# correlation of a single time series with a lagged copy of itself
# Often it is a lag of 1
# with financial time series, when returns have a negative autocorrelation, we say it is "mean reverting"
# if positive autocorrelation, we say it is "trend following"
# exchange rate example
df.index = pd.to_datetime(df.index)
df = df.resample(rule='M', how='last')
df['Return'] = df['Price'].pct_change()
autocorrelation = df['Return'].autocorr()
print("The autocorrelations is: ", autocorrelation)

# Exercises
# Convert the daily data to weekly data
MSFT = MSFT.resample(rule='W').last()

# Compute the percentage change of prices
returns = MSFT.pct_change()

# Compute and print the autocorrelation of returns
autocorrelation = returns['Adj Close'].autocorr()
print("The autocorrelation of weekly returns is %4.2f" %(autocorrelation))

# Compute the daily change in interest rates 
daily_diff = daily_rates.diff()

# Compute and print the autocorrelation of daily changes
autocorrelation_daily = daily_diff['US10Y'].autocorr()
print("The autocorrelation of daily interest rate changes is %4.2f" %(autocorrelation_daily))

# Convert the daily data to annual data
yearly_rates = daily_rates.resample(rule='A').last()

# Repeat above for annual data
yearly_diff = yearly_rates.diff()
autocorrelation_yearly = yearly_diff['US10Y'].autocorr()
print("The autocorrelation of annual interest rate changes is %4.2f" %(autocorrelation_yearly))

# Autocorrelation Function (ACF)
# can see which lags have the most autocorrelation
# good for finding a parsimonious model for fitting the data

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(x, lags=20, alpha=0.05) #x is a series or array
#lags = how many lags of the acf will be plotted
#alpha = width of the confidence interval
#alpha = 0.05 - 5% chance that if true autocorrelation is zero, it will fall 
    #outside blue band
#confidence bands are wider if alpha is lower or if you have fewer observations
#if you don't want to see confidence intervals, set alpha=1
#if you don't want to plot it, but see the numbers:
print(acf(x))

# Exercises
# Import the acf module and the plot_acf module from statsmodels
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

# Compute the acf array of HRB
acf_array = acf(HRB)
print(acf_array)

# Plot the acf function
plot_acf(HRB, alpha=1)
plt.show()

# Import the plot_acf module from statsmodels and sqrt from math
from statsmodels.graphics.tsaplots import plot_acf
from math import sqrt

# Compute and print the autocorrelation of MSFT weekly returns
autocorrelation = returns['Adj Close'].autocorr()
print("The autocorrelation of weekly MSFT returns is %4.2f" %(autocorrelation))

# Find the number of observations by taking the length of the returns DataFrame
nobs = len(returns)

# Compute the approximate confidence interval
conf = 1.96/sqrt(nobs)
print("The approximate confidence interval is +/- %4.2f" %(conf))

# Plot the autocorrelation function with 95% confidence intervals and 20 lags using plot_acf
plot_acf(returns, alpha=0.05, lags=20)
plt.show()

# White Noise
    # constant mean
    #constant variance
    # zero autocorrelations at all lags
# Special cases:
    # if data has normal distribution: Gaussian White Noise
noise = np.random.normal(loc=0, scale=1, size=500)
# loc = mean, scale = std dev
#autocorrelations of white noise series = 0

# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Simulate white noise returns
returns = np.random.normal(loc=0.02, scale=0.05, size=1000)

# Print out the mean and standard deviation of returns
mean = np.mean(returns)
std = np.std(returns)
print("The mean is %5.3f and the standard deviation is %5.3f" %(mean,std))

# Plot returns series
plt.plot(returns)
plt.show()

# Plot autocorrelation function of white noise returns
plot_acf(returns, lags=20)
plt.show()

#Random Walk
# Today's price = Yesterday's price + noise
# change in price is just random noise
# You can't forecast a random walk - best forecast for tomorrow's price
    #is today's price
# Random walk with drift - they drift by mu every period
# Change in price is still white noise with a mean of mu
# Statistical test for Random Walk
# regress the difference in prices on the lagged price, and test if it is zero
# this is called an augmented dickey-fuller test
results = adfuller(df['SPX'])
print(results[1])  #prints p value
#if p values is less than 5%, we can reject the null hypothesis

# Exercises:

# Generate 500 random steps with mean=0 and standard deviation=1
steps = np.random.normal(loc=0, scale=1, size=500)

# Set first element to 0 so that the first price will be the starting stock price
steps[0]=0

# Simulate stock prices, P with a starting price of 100
P = 100 + np.cumsum(steps)

# Plot the simulated stock prices
plt.plot(P)
plt.title("Simulated Random Walk")
plt.show()

# Generate 500 random steps
steps = np.random.normal(loc=0.001, scale=0.01, size=500) + 1

# Set first element to 1
steps[0]=1

# Simulate the stock price, P, by taking the cumulative product
P = 100 * np.cumprod(steps)

# Plot the simulated stock prices
plt.plot(P)
plt.title("Simulated Random Walk with Drift")
plt.show()

# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller

# Run the ADF test on the price series and print out the results
results = adfuller(AMZN['Adj Close'])
print(results)

# Just print out the p-value
print('The p-value of the test on prices is: ' + str(results[1]))

# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller

# Create a DataFrame of AMZN returns
AMZN_ret = AMZN.pct_change()

# Eliminate the NaN in the first row of returns
AMZN_ret = AMZN_ret.dropna()

# Run the ADF test on the return series and print out the p-value
results = adfuller(AMZN_ret['Adj Close'])
print('The p-value of the test on returns is: ' + str(results[1]))

#Stationarity
# Strong Stationarity - 
    # entire distribution of data is time-invariant (not dependent on time)
    # hard to test
#easier to test for weak stationarity
    # mean, variance, and autocorrelation are time invariant
# If a process is not stationary, then the parameters are different at each 
    #point in time, then there are too many parameters to estimate
# You might end up with more parameters than actual data
# Stationarity is necessary for a parsimonious model
# Random walk is a common type of non-staitonary series
# Seasonal series are non stationary
# Can transform nonstationary into stationary
    #take the first differences
SPY.diff()
#can get rid of specific seasonal difference by adding lag of 4
HRB.diff(4)
#can use log and diff together (bc of exponential growth and seasonality)
np.log(AMZN).diff(4)
#white noise is stationary

# Exercises:

# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Seasonally adjust quarterly earnings
HRBsa = HRB.diff(4)

# Print the first 10 rows of the seasonally adjusted series
print(HRBsa.head(10))

# Drop the NaN data in the first four rows
HRBsa = HRBsa.dropna()

# Plot the autocorrelation function of the seasonally adjusted series
plot_acf(HRBsa)
plt.show()

# AR Model
# Autoregressive model
#today's value = mean + fraction phi of yesterday's value + noise
# since it is only lagged by one this is called:
    #AR model of order 1 or AR(1) model
# If AR parameter phi =1 then the process is a random walk
# If phi =0 then the process is white noise
# for stationarity -1 < phi < 1
# if phi is negative, then a postive return last period at time t-1, 
    # implies that this periods return is more likely to be negative
    #aka mean reversion
# if phi is positive, then a positive return last period at time t-1,
    #implies that this periods returns is more likely to be positive
    #aka momentum
# AR(1) autocorrelation decays exponentially at a rate of phi
# Can help to work with simulated data to udnerstand AR process    

from statsmodels.tsa.arima_process import ArmaProcess
ar = np.array([1, -0.9])
ma = np.array([1])
AR_object = ArmaProcess(ar, ma)
simulated_data = AR_object.generate_sample(nsample=1000)
plt.plot(simulated_data)

#phi is positive, but we need the sin so phi=0.9, sin = -0.9

# Exercises:
# import the module for simulating data
from statsmodels.tsa.arima_process import ArmaProcess

# Plot 1: AR parameter = +0.9
plt.subplot(2,1,1)
ar1 = np.array([1, -0.9])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = AR_object1.generate_sample(nsample=1000)
plt.plot(simulated_data_1)

# Plot 2: AR parameter = -0.9
plt.subplot(2,1,2)
ar2 = np.array([1, 0.9])
ma2 = np.array([1])
AR_object2 = ArmaProcess(ar2, ma2)
simulated_data_2 = AR_object2.generate_sample(nsample=1000)
plt.plot(simulated_data_2)
plt.show()

# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Plot 1: AR parameter = +0.9
plot_acf(simulated_data_1, alpha=1, lags=20)
plt.show()

# Plot 2: AR parameter = -0.9
plot_acf(simulated_data_2, alpha=1, lags=20)
plt.show()

# Plot 3: AR parameter = +0.3
plot_acf(simulated_data_3, alpha=1, lags=20)
plt.show()

from statusmodels.tsa.arima_mmodel import ARMA 
mod = ARMA(simulated_data, order=(1,0))
result = mod.fit()
print(result.summary())
print(result.params)

# Forecasting an AR Model
mod = ARMA(simulated_data, order=(1,0))
res = mod.fit()
res.plot_predict(start='2016-07-01', end='2017-06-01')
plt.show()

# Exercises:
# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Fit an AR(1) model to the first simulated data
mod = ARMA(simulated_data_1, order=(1,0))
res = mod.fit()

# Print out summary information on the fit
print(res.summary())

# Print out the estimate for the constant and for phi
print("When the true phi=0.9, the estimate of phi (and the constant) are:")
print(res.params)

# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Forecast the first AR(1) model
mod = ARMA(simulated_data_1, order=(1,0))
res = mod.fit()
res.plot_predict(start=990, end=1010)
plt.show()

# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Forecast interest rates using an AR(1) model
mod = ARMA(interest_rate_data, order=(1,0))
res = mod.fit()

# Plot the original series and the forecasted series
res.plot_predict(start=0, end='2022')
plt.legend(fontsize=8)
plt.show()

# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Plot the interest rate series and the simulated random walk series side-by-side
fig, axes = plt.subplots(2,1)

# Plot the autocorrelation of the interest rate series in the top plot
fig = plot_acf(interest_rate_data, alpha=1, lags=12, ax=axes[0])

# Plot the autocorrelation of the simulated random walk series in the bottom plot
fig = plot_acf(simulated_data, alpha=1, lags=12, ax=axes[1])

# Label axes
axes[0].set_title("Interest Rate Data")
axes[1].set_title("Simulated Random Walk Data")
plt.show()

# PACF - Partial autocorrelation Function
# Lets you add another lag
# Phi  4,4, - PACF and it representats how significant adding a fourth lag is when you already have three lags
# similar python to acf
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(x, lags=20, alpha=0.5)
# x is the series or array, lags is the # of lags pacf to be plotted, alpha is the width of the conf interval

# Information Criteria - adjusts the goodness of fit for number of parameters (to avoid overfitting)
# Goodness of fit measures:
    # AIC (Akaike Information Criterion)
    # BIC (Bayesian Information Criterion)
from statsmodels.tsa.arima_mmodel import ARIMA 
mod = ARIMA(simulated_data, order=(1,0))
result = mod.fit()

result.summary()
result.params
result.aic 
result.bic 

# you want to pick one with the lowest information criterion

# Exercises

# Import the modules for simulating data and for plotting the PACF
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_pacf

# Simulate AR(1) with phi=+0.6
ma = np.array([1])
ar = np.array([1, -0.6])
AR_object = ArmaProcess(ar, ma)
simulated_data_1 = AR_object.generate_sample(nsample=5000)

# Plot PACF for AR(1)
plot_pacf(simulated_data_1, lags=20)
plt.show()

# Simulate AR(2) with phi1=+0.6, phi2=+0.3
ma = np.array([1])
ar = np.array([1, -0.6, -0.3])
AR_object = ArmaProcess(ar, ma)
simulated_data_2 = AR_object.generate_sample(nsample=5000)

# Plot PACF for AR(2)
plot_pacf(simulated_data_2, lags=20)
plt.show()

# Import the module for estimating an ARIMA model
from statsmodels.tsa.arima.model import ARIMA

# Fit the data to an AR(p) for p = 0,...,6 , and save the BIC
BIC = np.zeros(7)
for p in range(7):
    mod = ARIMA(simulated_data_2, order=(p,0,0))
    res = mod.fit()
# Save BIC for AR(p)    
    BIC[p] = res.bic
    
# Plot the BIC as a function of p
plt.plot(range(1,7), BIC[1:7], marker='o')
plt.xlabel('Order of AR Model')
plt.ylabel('Bayesian Information Criterion')
plt.show()

# MA - Moving Average Model
# Rt = mu + Eta t + Theta Eta t-1
# Today's noise = mean + Noise + fraction of yesterday's noise
# MA(1) - only one lagged error on right hand side so it's an MA of order 1
# If theta is 0, then the process is white noise
# Stationary for all values of theta
# Negative theta - one period mean reversion
# Postive theta - one period momentum
# lag 1 autocoreelation is not theta, but theta over 1 plus theta squared

from statsmodels.tsa.arima_process import ArmaProcess
ar = np.array([1])
ma = np.array([1,0.5])
AR_object = ArmaProcess(ar, ma)
simulated_data = AR_object.generate_sample(nsample=1000)
plt.plot(simulated_data)

# Don't need to reverse the sign of theta like you do with AR simulation

# Exercises:
# import the module for simulating data
from statsmodels.tsa.arima_process import ArmaProcess

# Plot 1: MA parameter = -0.9
plt.subplot(2,1,1)
ar1 = np.array([1])
ma1 = np.array([1, -0.9])
MA_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = MA_object1.generate_sample(nsample=1000)
plt.plot(simulated_data_1)

# Plot 2: MA parameter = +0.9
plt.subplot(2,1,2)
ar2 = np.array([1])
ma2 = np.array([1,0.9])
MA_object2 = ArmaProcess(ar2, ma2)
simulated_data_2 = MA_object2.generate_sample(nsample=1000)
plt.plot(simulated_data_2)

plt.show()


# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Plot 1: MA parameter = -0.9
plot_acf(simulated_data_1, lags=20)
plt.show()

# Plot 2: MA parameter = 0.9
plot_acf(simulated_data_2, lags=20)
plt.show()

# Plot 3: MA parameter = -0.3
plot_acf(simulated_data_3, lags=20)
plt.show()

# Estimation and Forecasting an MA Model

from statsmodels.tsa.arima_mmodel import ARIMA 
mod = ARIMA(simulated_data, order=(0,0,1))
result = mod.fit()

# Forecasting an MA model
from statsmodels.graphics.tsaplots import plot_predict
fig, ax = plt.subplots()
data.plot(ax=ax)
plot_predict(res, start='2012-09-27', end='2012-10-06', ax=ax)
plt.show()

# Exercises
# Import the ARIMA module from statsmodels
from statsmodels.tsa.arima.model import ARIMA

# Fit an MA(1) model to the first simulated data
mod = ARIMA(simulated_data_1, order=(0,0,1))
res = mod.fit()

# Print out summary information on the fit
print(res.summary())

# Print out the estimate for the constant and for theta
print("When the true theta=-0.9, the estimate of theta is:")
print(res.params[1])

# Import the ARIMA and plot_predict from statsmodels
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict

# Forecast the first MA(1) model
mod = ARIMA(simulated_data_1, order=(0,0,1))
res = mod.fit()

# Plot the data and the forecast
fig, ax = plt.subplots()
simulated_data_1.loc[950:].plot(ax=ax)
plot_predict(res, start=1000, end=1010, ax=ax)
plt.show()

# ARMA Models
# combo of AR & MA model
# Can be converted to pure AR or pure MA models

# Exercises:
# import datetime module
import datetime

# Change the first date to zero
intraday.iloc[0,0] = 0

# Change the column headers to 'DATE' and 'CLOSE'
intraday.columns = ['DATE', 'CLOSE']

# Examine the data types for each column
print(intraday.dtypes)

# Convert DATE column to numeric
intraday['DATE'] = pd.to_numeric(intraday['DATE'])

# Make the `DATE` column the new index
intraday = intraday.set_index('DATE')

# Notice that some rows are missing
print("If there were no missing rows, there would be 391 rows of minute data")
print("The actual length of the DataFrame is:", len(intraday))

# Everything
set_everything = set(range(391))

# The intraday index as a set
set_intraday = set(intraday.index)

# Calculate the difference
set_missing = set_everything - set_intraday

# Print the difference
print("Missing rows: ", set_missing)

# Fill in the missing rows
intraday = intraday.reindex(range(391), method='ffill')

# From previous step
intraday = intraday.reindex(range(391), method='ffill')

# Change the index to the intraday times
intraday.index = pd.date_range(start='2017-09-01 9:30', end='2017-09-01 16:00', freq='1min')

# Plot the intraday time series
intraday.plot(grid=True)
plt.show()

# From previous step
intraday = intraday.reindex(range(391), method='ffill')

# Change the index to the intraday times
intraday.index = pd.date_range(start='2017-09-01 9:30', end='2017-09-01 16:00', freq='1min')

# Plot the intraday time series
intraday.plot(grid=True)
plt.show()

# import the modules for simulating data and plotting the ACF
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf

# Build a list MA parameters
ma = [.8**i for i in range(30)]

# Simulate the MA(30) model
ar = np.array([1])
AR_object = ArmaProcess(ar, ma)
simulated_data = AR_object.generate_sample(nsample=5000)

# Plot the ACF
plot_acf(simulated_data, lags=30)
plt.show()


