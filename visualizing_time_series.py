# Visualizing Time Series
# Pandas refresh
import pandas as pd
df = read_csv('ch2_co2_levels.csv')
print(df.head(n=5))
print(df.dtypes)

# when working with time series, best to have date columns be datetime64 types
pd.to_datetime(['2009/07/31', 'test'])
# will return an NaT timestamp when the object cannot be parsed
pd.to_datetime(['2009/07/31', 'test'], errors='coerce')

# Exercises:

# Import pandas
import pandas as pd

# Read in the file content in a DataFrame called discoveries
discoveries = pd.read_csv(url_discoveries)

# Display the first five lines of the DataFrame
print(discoveries.head(n=5))

# Print the data type of each column in discoveries
print(discoveries.dtypes)

# Convert the date column to a datestamp type
discoveries['date'] = pd.to_datetime(discoveries['date'])

# Print the data type of each column in discoveries, again
print(discoveries.dtypes)

# Matplotlib
import matplotlib.pyplot as plt
# recommended to set date of time series as index
df = df.set_index('date_column')
df.plot()
plt.show()

# different templates
plt.style.use('fivethrityeight')
# should use labels
ax = df.plot(color='blue')
ax.set_xlabel('Date')
ax.set_ylabel('The values of my Y axis')
ax.set_title('The title of my plot')
plt.show()

# can set figure size, linewidth, linestyle, fontsize
ax = df.plot(figsize=(12,5), fontsize=12, linewidth=3, linestyle='--')
ax.set_xlabel('Date', fontsize=16)
ax.set_ylabel('The values of my Y axis', fontsize=16)
ax.set_title('The title of my plot', fontsize=16)
plt.show()

# Exercises

# Set the date column as the index of your DataFrame discoveries
discoveries = discoveries.set_index('date')

# Plot the time series in your DataFrame
ax = discoveries.plot(color='blue')

# Specify the x-axis label in your plot
ax.set_xlabel('Date')

# Specify the y-axis label in your plot
ax.set_ylabel('Number of great discoveries')

# Show plot
plt.show()

# Import the matplotlib.pyplot sub-module
import matplotlib.pyplot as plt

# Use the fivethirtyeight style
plt.style.use('fivethirtyeight')

# Plot the time series
ax1 = discoveries.plot()
ax1.set_title('FiveThirtyEight Style')
plt.show()

# Import the matplotlib.pyplot sub-module
import matplotlib.pyplot as plt

# Use the ggplot style
plt.style.use('ggplot')
ax2 = discoveries.plot()

# Set the title
ax2.set_title('ggplot Style')
plt.show()

# Plot a line chart of the discoveries DataFrame using the specified arguments
ax = discoveries.plot(color='blue', figsize=(8, 3), linewidth=2, fontsize=6)

# Specify the title in your plot
ax.set_title('Number of great inventions and scientific discoveries from 1860 to 1959', fontsize=8)

# Show plot
plt.show()

# Customize
# can slice time series data
discoveries['1950':'1970']
discoveries['1960-01':'1960-12']
discoveries['1950-01-01':'1950-01-15']

plt.style.use('fivethirtyeight')
df_subset = discoveries['1960':'1970']
ax = df_subset.plot(color='blue', fontsize=14)
plt.show()

# Can draw horizontal or vertical lines
ax.axvline(x='1969-01-01', color='red', linestyle='--')
ax.axhline(y=100, color='green', linestyle='--')

# can add shaded areas using axvspan or axhspan
ax.axvspan('1964-01-01', '1968-01-01', color='red', alpha=0.5)
ax.axhspan(8, 6, color='green', alpha=0.2)

# Exercises
# Select the subset of data between 1945 and 1950
discoveries_subset_1 = discoveries['1945-01-01':'1950-01-01']

# Plot the time series in your DataFrame as a blue area chart
ax = discoveries_subset_1.plot(color='blue', fontsize=15)

# Show plot
plt.show()

# Select the subset of data between 1939 and 1958
discoveries_subset_2 = discoveries['1939-01-01':'1958-01-01']

# Plot the time series in your DataFrame as a blue area chart
ax = discoveries_subset_2.plot(color='blue', fontsize=15)

# Show plot
plt.show()

# Plot your the discoveries time series
ax = discoveries.plot(color='blue', fontsize=6)

# Add a red vertical line
ax.axvline(x='1939-01-01', color='red', linestyle='--')

# Add a green horizontal line
ax.axhline(y=4, color='green', linestyle='--')

plt.show()

# Plot your the discoveries time series
ax = discoveries.plot(color='blue', fontsize=6)

# Add a vertical red shaded region
ax.axvspan('1900-01-01', '1915-01-01', color='red', alpha=0.3)

# Add a horizontal green shaded region
ax.axhspan(6, 8, color='green', alpha=0.3)

plt.show()

# Cleaning time series data
# finding missing values
print(df.isnull())
print(df.notnull())

print(df.isnull().sum())

df = df.fillna(method='bfill')

# Exercises

# Display first seven rows of co2_levels
print(co2_levels.head(n=7))

# Set datestamp column as index
co2_levels = co2_levels.set_index('datestamp')

# Print out the number of missing values
print(co2_levels.isnull().sum())

# Impute missing values with the next valid observation
co2_levels = co2_levels.fillna(method='bfill')

# Print out the number of missing values
print(co2_levels.isnull().sum())

# Plot aggregates of your data
# MOving average (aka rolling mean)
# Good for:
    # smoothing out short term fluctuations
    # removing outliers
    # highlighting long term trends or cycles

co2_levels_mean= co2_levels.rolling(window=52).mean()
ax = co2_levels_mean.plot()
ax.set_xlabel("Date")
ax.set_ylabel("Y axis")
ax.set_title("52 weeks rolling mean of time series")
plt.show()

index_month = co2_levels.index.month
co2_levels_by_month = co2_levels.groupby(index_month).mean()
co2_levels_by_month.plot()
plt.show()

# Exercises
# Compute the 52 weeks rolling mean of the co2_levels DataFrame
ma = co2_levels.rolling(window=52).mean()

# Compute the 52 weeks rolling standard deviation of the co2_levels DataFrame
mstd = co2_levels.rolling(window=52).std()

# Add the upper bound column to the ma DataFrame
ma['upper'] = ma['co2'] + (mstd['co2'] * 2)

# Add the lower bound column to the ma DataFrame
ma['lower'] = ma['co2'] - (mstd['co2'] * 2)

# Plot the content of the ma DataFrame
ax = ma.plot(linewidth=0.8, fontsize=6)

# Specify labels, legend, and show the plot
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('CO2 levels in Mauai Hawaii', fontsize=10)
ax.set_title('Rolling mean and variance of CO2 levels\nin Mauai Hawaii from 1958 to 2001', fontsize=10)
plt.show()

# Get month for each dates in the index of co2_levels
index_month = co2_levels.index.month

# Compute the mean CO2 levels for each month of the year
mean_co2_levels_by_month = co2_levels.groupby(index_month).mean()

# Plot the mean CO2 levels for each month of the year
mean_co2_levels_by_month.plot(fontsize=6)

# Specify the fontsize on the legend
plt.legend(fontsize=10)

# Show plot
plt.show()

# Summarizing values
df.describe()

# provides info on the shape, variability, and median of the data
ax1 = df.boxplot()

# histograms
ax2 = df.plot(kind='hist', bins=100)

# kernel density plots
# dampens effects of outliers and noise
ax3 = df.plot(kind='density', linewidth=2)

# Exercises
# Print out summary statistics of the co2_levels DataFrame
print(co2_levels.describe())

# Print out the minima of the co2 column in the co2_levels DataFrame
print(co2_levels['co2'].min())

# Print out the maxima of the co2 column in the co2_levels DataFrame
print(co2_levels['co2'].max())

# Generate a boxplot
ax = co2_levels.boxplot()

# Set the labels and display the plot
ax.set_xlabel('CO2', fontsize=10)
ax.set_ylabel('Boxplot CO2 levels in Maui Hawaii', fontsize=10)
plt.legend(fontsize=10)
plt.show()

# Generate a histogram
ax = co2_levels.plot(kind='hist', bins=50, fontsize=6)

# Set the labels and display the plot
ax.set_xlabel('CO2', fontsize=10)
ax.set_ylabel('Histogram of CO2 levels in Maui Hawaii', fontsize=10)
plt.legend(fontsize=10)
plt.show()

# Display density plot of CO2 levels values
ax = co2_levels.plot(kind='density', linewidth=4, fontsize=6)

# Annotate x-axis labels
ax.set_xlabel('CO2', fontsize=10)

# Annotate y-axis labels
ax.set_ylabel('Density plot of CO2 levels in Maui Hawaii', fontsize=10)

plt.show()

# Autocorrelation and partial autocorrelation
# how to detect and visualize seasonality, trend, and noise in time series
# Autocorelation = correlation between a time series and a delayed copy of itself
# used to find repetitive patterns or periodic signals

from statsmodels.graphics import tsaplots
fig = tsaplots.plot_acf(co2_levels['co2'], lag=40)
plt.show()

# blue shaded regions are areas of uncertainty

# partial autocorrelation measures the correlation coefficient between time series and lagged version of itself

fig = tsaplots.plot_pacf(co2_levels['co2'], lags=40)

# Exercises

# Import required libraries
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from statsmodels.graphics import tsaplots

# Display the autocorrelation plot of your time series
fig = tsaplots.plot_acf(co2_levels['co2'], lags=24)

# Show plot
plt.show()

# Import required libraries
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from statsmodels.graphics import tsaplots

# Display the partial autocorrelation plot of your time series
fig = tsaplots.plot_pacf(co2_levels['co2'], lags=24)

# Show plot
plt.show()

# Seasonality, trend and noise
