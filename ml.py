import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('timss.csv')
print(df)

df['belonging'].replace(999999, 0, inplace=True)
df['bullying'].replace(999999, 0, inplace=True)

df = df[(df['belonging'] != 0) & (df['bullying'] != 0)]

df = df.dropna()

print('The number of observed scores for Students Belonging scale:', len(df.belonging))
print('The number of observed scores for Students School Bullying Experience scale:', len(df.bullying))
print(df[['belonging', 'bullying']].head(5515))

y = df['belonging']
print(y)

x = df.drop('belonging', axis=1)
print(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

print(y_lr_train_pred)
print(y_lr_test_pred)

from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print('LR MSE (Train):', lr_train_mse)
print('LR R2 (Train):', lr_train_r2)
print('LR MSE (Test):', lr_test_mse)
print('LR R2 (Test):', lr_test_r2)

lr_results = pd.DataFrame(['Linear Regression Results:', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Testing MSE', 'Testing R2']
print(lr_results)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest Results:', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Testing MSE', 'Testing R2']
print(rf_results)

df_model = pd.concat([lr_results, rf_results], axis=0)
print(df_model)

plt.figure(figsize=(5,5))
plt.scatter(x=y_train, y=y_lr_train_pred, c='#7CAE00', alpha=0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)


plt.plot(y_train, p(y_train), c='#F8766D')
plt.ylabel('Predict Logs')
plt.xlabel('Experimental Logs')
plt.show()