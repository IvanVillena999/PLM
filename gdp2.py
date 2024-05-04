import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#Read the GDP time series data
df = pd.read_csv('gdp.csv')
print(df.head(90))

print('\nThe number of observations is:', len(df))

#Indexing the time variable (year, quarter)
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d')

df.index = df['time']
del df['time']
print(df.head())

#Line graph for GDP
sns.lineplot(df)
plt.xlabel('Quarters')
plt.ylabel('GDP per Quarter (in Million PHP)')
plt.show()

#Compute for rolling mean, rolling std
rolling_mean = df.rolling(7).mean()
rolling_std = df.rolling(7).std()

#Visualize the rm and rstd thru line plot
plt.plot(df, color='blue', label='Original GDP Data')
plt.plot(rolling_mean, color='red', label='Rolling Mean GDP')
plt.plot(rolling_std, color='black', label='Rolling STD GDP')
plt.title('GDP Time Series Data, Rolling Mean GDP and Rolling STD')
plt.legend(loc='best')
plt.show()

#Compute for ADF for Stationarity (1, 3, 6, 9 mos lag)
from statsmodels.tsa.stattools import adfuller
adft = adfuller(df, autolag='AIC')

output_df = pd.DataFrame({'Values': [adft[0], adft[1], adft[2], adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']], 'Metric':['Test Statistics', 'p-value', 'No of lags used',
                                                                                                                                'No of observations used', 'critical value at 1%', 'critical value at 5%', 'critical value at 10%']})
print(output_df) #Non-stationary since Test statistics > critical values 1%, 5%, 10% and p-value is > 0.05
#Why GDP data us non-stationary?

#Detect autocorrelation
autocorr_lag1 = df['gdp'].autocorr(lag=1)
print('\nOne-month lag:', autocorr_lag1)

autocorr_lag3 = df['gdp'].autocorr(lag=3)
print('\n3-month lag:', autocorr_lag3)

autocorr_lag6 = df['gdp'].autocorr(lag=6)
print('\n6-month lag:', autocorr_lag6)

autocorr_lag9 = df['gdp'].autocorr(lag=9)
print('\n9-month lag:', autocorr_lag9)

#Highly correlated at all lags, therefore, lets analyze the trend and pattern (time series only)
#Decomposition to visualize trends and patterns in GDP (time series)
from statsmodels.tsa.seasonal import seasonal_decompose
decompose = seasonal_decompose(df['gdp'], model='additive', period=7)
decompose.plot()
plt.show()

#Forecast values
df['time'] = df.index #Training datasets (used to forecast values)
train = df[df['time'] < pd.to_datetime('2022-07-01', format='%Y-%m-%d')].copy()
train['train'] = train['gdp']
train =train.drop(columns=['time', 'gdp'])

#Testing set
test = df[df['time'] >= pd.to_datetime('2022-07-01', format='%Y-%m-%d')].copy()
test['test'] = test['gdp']
test = test.drop(columns=['time', 'gdp'])

#Visualize the training and testing dataset
plt.plot(train, color='black')
plt.plot(test, color='red')
plt.title('Train and Test split for GDP Data')
plt.xlabel('GDP Data')
plt.ylabel('Quarters')
sns.set()
plt.show()

from pmdarima.arima import auto_arima
train.index = pd.to_datetime(train.index)
test.index = pd.to_datetime(test.index) #ARIMA Model

#Build the ARIMA Model
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)

#Forecast values
forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast, columns=['Prediction'])

forecast.index = test.index

#Visualize the predicted values
plt.figure(figsize=(10, 6))
plt.plot(train, label='Training Data', color='blue')
plt.plot(test, label='Testing Data', color='green')
plt.plot(forecast, label='Forecast', color='red')

plt.title('GDP Forecast using ARIMA')
plt.xlabel('Quarters')
plt.ylabel('GDP Data')
plt.legend()
plt.grid(True)
plt.show()

#Test the accuracy of the model using RMSE (root mean squared error)
from math import sqrt
from sklearn.metrics import mean_squared_error
rms = sqrt(mean_squared_error(test, forecast))
print('RMSE:', rms)

#If RMSE is small, the predicted value is accurate, otherwise.