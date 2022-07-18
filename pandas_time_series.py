# Using dates with Pandas
# Objects for points in time and time periods
# Time can be a series or DF columns, but a DF Index converts the object into Time Series
# Basic building block:
    # pd.Timestamp
import pandas as pd
import numpy as np
from datetime import datetime
time_stamp = pd.Timestamp(datetime(2017, 1, 1))

# can also use a date string instead of an object
pd.Timestamp('2017-01-01') == time_stamp
time_stamp.year
time_stamp.day_name()

period = pd.Period('2017-01')
# Period defaults to a month
period.asfreq('D')
period + 2  #(returns 2 months ahead)
pd.Timestamp('2017-01-01', 'M') + 1

# time series requires a sequence of dates
index = pd.date_range(start='2018-01-01', periods=12, freq='M')
# can also specify an enddate instead of number of periods
# default is daily frequency
pd.DateTimeIndex    #sequence of Timestamp objects with frequency info
# can index into it   index[0]

pd.DataFrame({'data': index}).info()

# data = np.random.random((size=12,2))
# pd.DataFrame(data=data,  index=index).info()
# creates time series with 12 monthly timestamps

# Exercises:

# Create the range of dates here
seven_days = pd.date_range(start='2017-1-1', periods=7)

# Iterate over the dates and print the number and name of the weekday
for day in seven_days:
    print(day.dayofweek, day.day_name())

# Transforming time series
google = pd.read_csv('google.csv')
google.info()

# to convert strings to datetime64
pd.to_datetime()

# date into index   .set_index()
google.set_index('date', inplace=True)

google.price.plot(title='Google Stock Price')
plt.tight_layout(); plt.show()

# selecting/indexing using strings to parse to dates
google['2015'].info()
google['2014-3': '2016-2'].info()
google.loc['2016-6-1', 'price']     #need to use full date and column label with .loc

# to set frequency
google.asfreq('D').info()   #set calendar day freq
google.asfreq('B').info()   #set as business day freq

# Exercises:
data = pd.read_csv('nyc.csv')

# Inspect data
print(data.info())

# Convert the date column to datetime64
data.date = pd.to_datetime(data.date)

# Set date column as index
data.set_index('date', inplace=True)

# Inspect data 
print(data.info())

# Plot data
data.plot(subplots=True)
plt.show()

data = pd.read_csv('nyc.csv')

# Inspect data
print(data.info())

# Convert the date column to datetime64
data.date = pd.to_datetime(data.date)

# Set date column as index
data.set_index('date', inplace=True)

# Inspect data 
print(data.info())

# Plot data
data.plot(subplots=True)
plt.show()

# Inspect data
print(co.info())

# Set the frequency to calendar daily
co = co.asfreq('D')

# Plot the data
co.plot(subplots=True)
plt.show()

# Set frequency to monthly
co = co.asfreq('M')

# Plot the data
co.plot(subplots=True)
plt.show()

google = pd.read_csv('google.csv', parse_dates=['date'], index_col='date')

# moving data between past and future
# defaults to periods=1
google['shifted'] = google.price.shift()
google['lagged'] = google.price.shift(periods=-1)

# to calculate one period percent change (called financial return in finance)
# returns relative change in price in percent
google['change'] = google.price.div(google.shifted)

google['return'] = google.change.sub(1).mul(100)

# .diff calculates the change between values at different points in time
# can be used to calculate one period returns
google['diff'] = google.price.diff()

# percent change for two adjacent periods
google['pct_change']  = google.price.pct_change().mul(100)

# default periods = 1
google['return_3d']  = google.price.pct_change(periods=3).mul(100)

# Import data here
google = pd.read_csv('google.csv', parse_dates=['Date'], index_col='Date')

# Set data frequency to business daily
google = google.asfreq('B')

# Create 'lagged' and 'shifted'
google['lagged'] = google.Close.shift(periods=-90)
google['shifted'] = google.Close.shift(periods=90)

# Plot the google price series
google.plot('lagged', 'shifted')
plt.show()

# Created shifted_30 here
yahoo['shifted_30'] = yahoo.price.shift(periods=30)

# Subtract shifted_30 from price
yahoo['change_30'] = yahoo.price.sub(yahoo.shifted_30)

# Get the 30-day price difference
yahoo['diff_30'] = yahoo.price.diff(periods=30)

# Inspect the last five rows of price
print(yahoo.tail(5))

# Show the value_counts of the difference between change_30 and diff_30
print(yahoo.change_30.sub(yahoo.diff_30).value_counts())

# Create daily_return
google['daily_return'] = google.Close.pct_change().mul(100) 

# Create monthly_return
google['monthly_return'] = google.Close.pct_change(periods=30).mul(100) 

# Create annual_return
google['annual_return'] = google.Close.pct_change(periods=360).mul(100) 

# Plot the result
google.plot(subplots=True)
plt.show()

# Compare growth rates
# stock prices: hard to compare at different levels
# one solution: normalize price series to start at 100
    # divide all time series by its first value and multiply by 100
first_price = google.price.iloc[0]
normalized = google.price.div(first_price).mul(100)
normalized.plot(title="Google Normalized Series")

# normalizing multiple series
prices = pd.read_csv('stock_prices.csv', parse_dates=['date'], index_col='date')
normalized = prices.div(prices.iloc[0])

index = pd.read_csv('benchmark.csv', parse_dates=['date'], index_col='date')
prices = pd.concat([prices, index], axis=1).dropna()

normalized = prices.div(prices.iloc[0]).mul(100)
normalized.plot()

# plotting performance difference
diff = normalized[tickers].sub(normalized['SP500'], axis=0)

# Exercises:
# Import data here
prices = pd.read_csv('asset_classes.csv', parse_dates=['DATE'], index_col='DATE')

# Inspect prices here
print(prices.info())

# Select first prices
first_prices = prices.iloc[0]

# Create normalized
normalized = prices.div(first_prices).mul(100)

# Plot normalized
normalized.plot()
plt.show()

# Import stock prices and index here
stocks = pd.read_csv('nyse.csv', parse_dates=['date'], index_col='date')
dow_jones = pd.read_csv('dow_jones.csv', parse_dates=['date'], index_col='date')

# Concatenate data and inspect result here
data = pd.concat([stocks, dow_jones], axis=1)
print(data.info())

# Normalize and plot your data here
normalized = data.div(data.iloc[0]).mul(100).plot()
plt.show()

# Create tickers
tickers = ['MSFT', 'AAPL']

# Import stock data here
stocks = pd.read_csv('msft_aapl.csv', parse_dates=['date'], index_col='date')

# Import index here
sp500 = pd.read_csv('sp500.csv', parse_dates=['date'], index_col='date')

# Concatenate stocks and index here
data = pd.concat([stocks, sp500], axis=1).dropna()

# Normalize data
normalized = data.div(data.iloc[0]).mul(100)

# Subtract the normalized index from the normalized stock prices, and plot the result
diff = normalized[tickers].sub(normalized['SP500'], axis=0).plot()
plt.show()

# Changing time series frequency: resampling
# frequency conversion affects the data
# upsampling creates new rows that you need to tell pandas how to fill or interpolate
# downsampling - you need to tell pandas how to aggregate existing data

dates = pd.date_range(start='2016', periods=4, freq='Q')
data = range(1,5)
quarterly = pd.Series(data=data, index=dates)

# default for quarterly is Dec for end of 4th quarter
monthly = quarterly.asfreq('M')

# ways to fill in missing values
monthly = monthly.to_frame('baseline')
monthly['ffill'] = quarterly.asfreq('M', method='ffill')   #forward fill
monthly['bfill'] = quarterly.asfreq('M', method='bfill')    #backward fill
monthly['value'] = quarterly.asfreq('M', fill_value=0)      #replace with a value

# you can reindex too
dates = pd.date_range(start='2016', periods=12, freq='M')
quarterly.reindex(dates)

# exercises
# Set start and end dates
start = '2016-1-1'
end = '2016-2-29'

# Create monthly_dates here
monthly_dates = pd.date_range(start=start, end=end, freq='M')

# Create and print monthly here
monthly = pd.Series(data=[1,2], index=monthly_dates)
print(monthly)

# Create weekly_dates here
weekly_dates = pd.date_range(start=start, end=end, freq='W')

# Print monthly, reindexed using weekly_dates
print(monthly.reindex(weekly_dates))
print(monthly.reindex(weekly_dates, method='bfill'))
print(monthly.reindex(weekly_dates, method='ffill'))

# Import data here
data = pd.read_csv('unemployment.csv', parse_dates=['date'], index_col='date')

# Show first five rows of weekly series
print(data.asfreq('W').head())

# Show first five rows of weekly series with bfill option
print(data.asfreq('W', method='bfill').head())

# Create weekly series with ffill option and show first five rows
weekly_ffill = data.asfreq('W', method='ffill')
print(weekly_ffill.head())

# Plot weekly_fill starting 2015 here 
weekly_ffill.loc['2015':].plot()
plt.show()


# Upsampling and interpoliation with .resample()
# works similar to groupby()
# groups data within resampling period and applies one or more methods to each group
# can be used to fill or interpolate values during upsampling
# or for applying aggregation for downsampling

unrate = pd.read_csv('unrate.csv', parse_dates['Date'], index_col='Date')

# resampling creates new date for frequency offset
# resample can assign a frequency if none exists 
# Calendar Month End    'M' 
# Calendar Month Start  'MS' 
# Business Month End    'BM'    may deviate based on weekends or holidays
# Business Month Start  'BMS' 

unrate.asfreq('MS').info()
unrate.resample('MS')   #creates a Resampler object
unrate.asfreq('MS').equals(unrate.resample('MS').asfreq())  #returns true

# resample only returns data when calling another method
gdp_1 = gdp.resample('MS').ffill().add_suffix('ffill')
gdp_2 = gdp.resample('MS').interpolate().add_suffix('_inter')
df1 = pd.DataFrame([1,2,3], columns=['df1'])
df2 = pd.DataFrame([4,5,6], columns=['df2'])
pd.concat([df1, df2])
pd.concat([df1, df2], axis=1)  #concats the DF's horizontally, alinging the row index

# Exercises:
# Inspect data here
print(monthly.info())

# Create weekly dates
weekly_dates = pd.date_range(start=monthly.index.min(), end=monthly.index.max(), freq='W')

# Reindex monthly to weekly data
weekly = monthly.reindex(weekly_dates)

# Create ffill and interpolated columns
weekly['ffill'] = weekly.UNRATE.ffill()
weekly['interpolated'] = weekly.UNRATE.interpolate()

# Plot weekly
weekly.plot()
plt.show()

# Import & inspect data here
data = pd.read_csv('debt_unemployment.csv', parse_dates=['date'], index_col='date')
print(data.info())

# Interpolate and inspect here
interpolated = data.interpolate()
print(interpolated.info())

# Plot interpolated data here
interpolated.plot(secondary_y='Unemployment')
plt.show()

# downsampling (day to month, hour to day)
# how to summarize existing data
ozone = ozone.resample('D').asfreq()
ozone.reample('M').mean().head()    #creates monthly freq with default month-end offset with monthly average

# can use more than one agg method
ozone.resample('M').agg(['mean', 'std']).head()

ozone = ozone.loc['2016':]
ax = ozone.plot()
monthly = ozone.resample('M').mean()
monthly.add_suffix('_monthly').plot(ax=ax)

data = pd.read_csv('ozone_pm25.csv', parse_dates=['date'], index_col='date')
data = data.resample('D').asfreq()

data = data.resample('BM').mean()

df.resample('M').first().head(4)
df.resample('MS').first().head(4)

# Exercises
# Import and inspect data here
ozone = pd.read_csv('ozone.csv', parse_dates=['date'], index_col='date')
ozone.info()


# Calculate and plot the weekly average ozone trend
ozone.resample('W').mean().plot()
plt.show()

# Calculate and plot the monthly average ozone trend
ozone.resample('M').mean().plot()
plt.show()

# Calculate and plot the annual average ozone trend
ozone.resample('A').mean().plot()
plt.show()

# Import and inspect data here
stocks = pd.read_csv('stocks.csv', parse_dates=['date'], index_col='date')
print(stocks.info())

# Calculate and plot the monthly averages
monthly_average = stocks.resample('M').mean()
monthly_average.plot(subplots=True)
plt.show()

# Import and inspect gdp_growth here
gdp_growth = pd.read_csv('gdp_growth.csv', parse_dates=['date'], index_col='date')
gdp_growth.info()

# Import and inspect djia here
djia = pd.read_csv('djia.csv', parse_dates=['date'], index_col='date')
djia.info()

# Calculate djia quarterly returns here 
djia_quarterly = djia.resample('QS').first()
djia_quarterly_return = djia_quarterly.pct_change().mul(100)

# Concatenate, rename and plot djia_quarterly_return and gdp_growth here 
data = pd.concat([gdp_growth, djia_quarterly_return], axis=1)
data.columns = ['gdp', 'djia']

data.plot()
plt.show()

# Import data here
sp500 = pd.read_csv('sp500.csv', parse_dates=['date'], index_col='date')
sp500.info()

# Calculate daily returns here
daily_returns = sp500.squeeze().pct_change()

# Resample and calculate statistics
stats = daily_returns.resample('M').agg(['mean', 'median', 'std'])

# Plot stats here
stats.plot()
plt.show()

