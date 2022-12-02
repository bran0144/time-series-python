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