### Developed No: Anbuselvan.S
### Register No: 212223240008
### Date:

# Ex.No: 6 - HOLT WINTERS EXPONENTIAL SMOOTHING METHOD FOR YAHOO STOCK PREDICTION.

## AIM:
Analyze and forecast yahoo stock prediction data using the Holt-Winters Exponential Smoothing mode.

## ALGORITHM:
1. You import the necessary libraries.
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, and perform some initial data exploration.
3. You group the data by date and resample it to a monthly frequency (beginning of the month.
4. You plot the time series data.
5. You import the necessary 'statsmodels' libraries for time series analysis.
6. You decompose the time series data into its additive components and plot them.
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance.
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-Winters model to the entire dataset and make future predictions.
9. You plot the original sales data and the predictions.

## PROGRAM:
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv('/content/yahoo_stock.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

Volume = df['Volume'].resample('MS').sum()

train = Volume[:-12]
test = Volume[-12:]

hw_model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12).fit()

test_predictions = hw_model.forecast(steps=12)

plt.figure(figsize=(10,6))
plt.plot(Volume, label='Original Yahoo stock Data')
plt.plot(test.index, test_predictions, label='Test Predictions', linestyle='--')
plt.title('Yahoo Stock Test Predictions')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.show()

print(test_predictions)

rmse = np.sqrt(mean_squared_error(test, test_predictions))
print(f'Test RMSE: {rmse:.2f}')

final_model = ExponentialSmoothing(Volume, seasonal='add', seasonal_periods=12).fit()
final_predictions = final_model.forecast(steps=12)

plt.figure(figsize=(10,6))
plt.plot(Volume, label='Original Yahoo Stock Data')
plt.plot(pd.date_range(Volume.index[-1], periods=12, freq='MS'), final_predictions, label='Final Predictions', linestyle='--')
plt.title('Yahoo Stock Final Predictions')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.show()

print(final_predictions)
```
## OUTPUT:

### TEST_PREDICTION - PLOT AND VALUES:
![image](https://github.com/user-attachments/assets/11fc0c5d-4162-45d5-b5b8-12e91bcd92a3)
![image](https://github.com/user-attachments/assets/07b7999d-feb0-457c-a2ea-aa037dba960d)

### FINAL_PREDICTION - PLOT AND VALUES:
![image](https://github.com/user-attachments/assets/0927a464-6433-4ce7-b348-5facab48bf5b)
![image](https://github.com/user-attachments/assets/db7438cd-6e10-448d-a1d1-1bd9c6c5d809)

## RESULT:
Thus the program run successfully based on the Holt Winters Method model for yahoo stock prediction.
