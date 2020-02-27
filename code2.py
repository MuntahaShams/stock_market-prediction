import numpy as np
import pandas as pd
import quandl
import datetime
import matplotlib.pyplot as plt

# Set start and end date for stock prices
start_date = datetime.date(2009, 3,8)
end_date = datetime.date.today()
data = quandl.get('FSE/SAP_X', start_date=start_date, end_date=end_date)# Load data from Quandl
data.to_csv('stock_market.csv')# Save data to CSV file
df = pd.DataFrame(data, columns=['Close'])# Create a new DataFrame with only closing price and date
df = df.reset_index()# Reset index column so that we have integers to represent time 

# Import matplotlib package for date plots
import matplotlib.dates as mdates
years = mdates.YearLocator() # Get every year
yearsFmt = mdates.DateFormatter('%Y') # Set year format
fig,ax= plt.subplots()# Create subplots to plot graph and control axes
ax.plot(df['Date'], df['Close'])
# Format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)

plt.title('Close Stock Price History [2009 - 2019]')
plt.xlabel('Date')
plt.ylabel('Closing Stock Price in $')
plt.show()

# Import package for splitting data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.20,random_state=0)

# Reshape index column to 2D array for .fit() method
X_train = np.array(train.index).reshape(-1, 1)
y_train = train['Close']

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

#Train set graph
plt.title('Linear Regression | Price vs Time')
plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
plt.plot(X_train, model.predict(X_train), color='r', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Create test arrays
X_test = np.array(test.index).reshape(-1, 1)
y_test = test['Close']

# Generate array with predicted values
y_pred = model.predict(X_test)
print("predicted stock market price of year are:",model.predict([[2026]]))
print("predicted stock market price of year are:",model.predict([[2052]]))
