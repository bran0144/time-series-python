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
