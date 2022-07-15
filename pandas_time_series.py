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

