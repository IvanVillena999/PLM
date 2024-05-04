import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy.stats


df = pd.read_csv('quanti.csv')
print(df.head(300))

x = df['hours']
y = df['spend']

df.plot(kind='scatter', x='hours', y='spend', figsize=(9, 9), color='black')
plt.xlabel('Number of Hours Spent Watching Videos (X)')
plt.ylabel('Amount Spent on Purchasing Products (Y)')
plt.title('Scatter Plot of Number of Hours Watching Videos and Amount Spent Puchasing Products', fontsize=12)
m, b = np.polyfit(df['hours'], df['spend'], 1)
plt.plot (x, m*x + b, color='red')
plt.show()


r, p = scipy.stats.pearsonr(x, y)
print('\nThe pearson coeffecient is:', r)
print('\nThe p-value of the pearson coeffecient is:', p)

if p < 0.5:
    print('\nThe null hypothesis is rejected')
if p > 0.5:
    print('\nThe null hypothesis is accepted')

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X=pd.DataFrame(df['hours']), y=df['spend'])

print('\nThe intercept of the model is:',model.intercept_ )
print('\nThe coefficient of the model is:',model.coef_)


print('\nAmount Spent = ', model.intercept_, '+', 'Hours Watching Videos', model.coef_)
print('A', model.coef_, 'increase in hours watching videos will increase by at least 1% the amount of money spent in purchasing products')


print('\nThe R-squared of the model is:', model.score(X=pd.DataFrame(df['hours']), y=df['spend']))


train_prediction = model.predict(X=pd.DataFrame(df['hours']))

residuals = df['spend'] - train_prediction
print(residuals.describe())


ssr = (residuals**2).sum()
ssrtotal = ((df['spend'] - df['spend'].mean())**2).sum()

r2 = 1 - ssr / ssrtotal
print('\nThe R-squared of the model is:', r2)


df.plot(kind='scatter', x='hours', y='spend', figsize=(9, 9), color='black', xlim=(0,7))
plt.plot(df['hours'], train_prediction, color='blue')
plt.show()


plt.figure(figsize=(9, 9))
stats.probplot(residuals, dist='norm', plot=plt)
plt.show()


from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(train_prediction, df['spend'])**0.5
print('\nThe RMSE of the model is:', rmse)

pred_value = 1026.0254193362741 + 2.69494861
print(pred_value)



