import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

df = pd.read_csv('gdp.csv')

df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d')

df.index = df['time']
del df['time']
print(df.head())

sns.lineplot(df)
plt.xlabel('Quarter')
plt.ylabel('GDP at 2018 Constant Prices (in Millions PHP)')
plt.show()

rolling_mean = df.rolling(7).mean()
rolling_std = df.rolling(7).std()

plt.plot(df, color='blue', label='Original GDP Data')
plt.plot(rolling_mean, color='red', label='Rolling Mean GDP Data')
plt.plot(rolling_std, color='black', label='Rolling Std GDP Data')
plt.title('GDP Time Series, Rolling Mean GDP Data, Rolling Std GDP Data')
plt.legend(loc='best')
plt.show()

from statsmodels.tsa.stattools import adfuller
adft = adfuller(df, autolag='AIC')

output_df = pd.DataFrame({"Values":[adft[0],adft[1],adft[2],adft[3], adft[4]['1%'], adft[4]['5%'], adft[4]['10%']]  , "Metric":["Test Statistics","p-value","No. of lags used","Number of observations used",
                                                        "critical value (1%)", "critical value (5%)", "critical value (10%)"]})
print(output_df)

autocorr_lag1 = df['gdp'].autocorr(lag=1)
print('\nOne-month lag:', autocorr_lag1)

autocorr_lag3 = df['gdp'].autocorr(lag=3)
print('\n3-month lag:', autocorr_lag3)

autocorr_lag6 = df['gdp'].autocorr(lag=6)
print('\n6-month lag:', autocorr_lag6)

autocorr_lag9 = df['gdp'].autocorr(lag=9)
print('\n9-month lag:', autocorr_lag9)

from statsmodels.tsa.seasonal import seasonal_decompose
decompose = seasonal_decompose(df['gdp'], model='additive', period=7)
decompose.plot()
plt.show()


df['time'] = df.index
train = df[df['time'] < pd.to_datetime("2022-07-01", format='%Y-%m-%d')].copy()
train['train'] = train['gdp']
train = train.drop(columns=['time', 'gdp'])

test = df[df['time'] >= pd.to_datetime("2022-07-01", format='%Y-%m-%d')].copy()
test['test'] = test['gdp']
test = test.drop(columns=['time', 'gdp'])

plt.plot(train, color='black')
plt.plot(test, color='red')
plt.title("Train/Test split for GDP Data")
plt.ylabel("Passenger Number")
plt.xlabel('Year-Month')
sns.set()
plt.show()

from pmdarima.arima import auto_arima

train.index = pd.to_datetime(train.index)
test.index = pd.to_datetime(test.index)

model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)

forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast, columns=['Prediction'])

forecast.index = test.index

plt.figure(figsize=(10, 6))
plt.plot(train, label='Training Data', color='blue')
plt.plot(test, label='Test Data', color='green')
plt.plot(forecast, label='Forecast', color='red')

plt.title('GDP Forecast with ARIMA')
plt.xlabel('Time')
plt.ylabel('GDP')
plt.legend()
plt.grid(True)
plt.show()

from math import sqrt
from sklearn.metrics import mean_squared_error
rms = sqrt(mean_squared_error(test,forecast))
print("RMSE: ", rms)