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

# Autocorrelation Function
