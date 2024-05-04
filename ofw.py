import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import seaborn as sns
import scipy.stats as stats

#Read datasets (ofw and unemployment)
df = pd.read_csv('unemp.csv')
print(df.head(20))

#1st step: scatterplot (x=unemployment, y=ofw)
df.plot(kind='scatter', x='unemp', y='ofw', figsize=(9, 9), color='black')
plt.show()

#2nd step: Linear regression model creation
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X=pd.DataFrame(df['unemp']), y=df['ofw'])

print('\nThe intercept of the model is:',model.intercept_ )
print('\nThe coefficient of the model is:',model.coef_)

#Linear Rgeression Model
print('\nOFW = ', model.intercept_, '+', 'unemployment', model.coef_)

#R-squared
print('\nThe R-squared of the model is:', model.score(X=pd.DataFrame(df['unemp']), y=df['ofw']))

#Predict the 2024 data for OFW
train_prediction = model.predict(X=pd.DataFrame(df['unemp']))
#Residuals: Actual - Predicted
residuals = df['ofw'] - train_prediction
print(residuals.describe())

#Residuals
ssr = (residuals**2).sum()
ssrtotal = ((df['ofw'] - df['ofw'].mean())**2).sum()

r2 = 1 - ssr / ssrtotal
print('\nThe R-squared of the model is:', r2)

#Scatterplot with regression line
df.plot(kind='scatter', x='unemp', y='ofw', figsize=(9, 9), color='black', xlim=(0,7))
plt.plot(df['unemp'], train_prediction, color='blue')
plt.show()

#Normalized the data
plt.figure(figsize=(9, 9))
stats.probplot(residuals, dist='norm', plot=plt)
plt.show()

#Compute for residiuals mean squared error (rmse)
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(train_prediction, df['ofw'])**0.5
print('\nThe RMSE of the model is:', rmse)

#Predict value for OFW (Y)
pred_value = 130.55001037822333 + (12.71208528**18.73)
print('\nThe predicted rate for OFW Deployment for 2024 is:', pred_value)

#Insert the predicted value to the exisitng dataset
df = pd.read_csv('unemp.csv')
current = pd.DataFrame({'year':2024, 'ofw': [pred_value]})
df = pd.concat([df, current], ignore_index=True)
print(df.head(20))

#END

#Why increase?
#Lack of opportunity
#Increasing USD exchange rate
#Social media influence



