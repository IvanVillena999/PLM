import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.stats as stats
matplotlib.style.use('ggplot')

df = pd.read_csv('unemp.csv')
print(df.head(19))

df.plot(kind='scatter', x='unemp', y='ofw', figsize=(9,9), color='black')
plt.show()

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X=pd.DataFrame(df['unemp']), y=df['ofw'])

print(model.intercept_)
print(model.coef_)

print('\nThe model fit line is:', 'ofw =', model.intercept_, '+', model.coef_, '*unemp')
print('\nR-squared:', model.score(X=pd.DataFrame(df['unemp']), y=df['ofw']))

train_prediction = model.predict(X=pd.DataFrame(df['unemp']))
residuals = df['ofw'] - train_prediction
print(residuals.describe())

ssr = (residuals**2).sum()
sstotal = ((df['ofw'] - df['ofw'].mean())**2).sum()

r2 = 1 - (ssr/sstotal)
print('\nR-squared:', r2)

df.plot(kind='scatter', x='unemp', y='ofw', figsize=(9,9), color='black', xlim=(0,7))
plt.plot(df['unemp'], train_prediction, color='blue')
plt.show()

plt.figure(figsize=(9,9))
stats.probplot(residuals, dist='norm', plot=plt)
plt.show()

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(train_prediction, df['ofw'])**0.5
print(rmse)

pred_value = (130.55001037822333 + (12.71208528**18.73))
print(pred_value)

df = pd.read_csv('unemp.csv')
current = pd.DataFrame({'year': [2024], 'ofw': [pred_value]})
df = pd.concat([df, current], ignore_index=True)
print(df.head(20))

