import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('freddata.csv')
print(df.head(10))

#Bar graph in seaborn
sns.barplot(df, x='days', y='fredtips', estimator='sum', errorbar=None, color='green')
plt.xlabel('Ferbuary 2024', fontsize=14)
plt.ylabel('Amount of Tips Received by Fred', fontsize=14)
plt.show()

#box and whiskers plot (5 summary measures: mean,min, max, 25% and 75%) matplotlib
box = df['fredtips']
fig = plt.figure(figsize=(10, 7))
plt.boxplot(box)
plt.xlabel('Amount of Tips Received by Fred')
plt.show()

#BW plot in seaborn
sns.boxplot(x='fredtips', data=df, color='green')
plt.xlabel('Amount of Tips', fontsize=14)
plt.show()

#Line graph matplot (if data are expressed in indeces/percentages, to show changes, determine trends
x = df['days']
y = df['fredtips']

plt.plot(x, y, color='green')
plt.xlabel('Days', fontsize=14)
plt.ylabel('Amount of Tips', fontsize=14)
plt.show()

#Line graph in seaborn
sns.lineplot(x='days', y='fredtips', data=df, color='black', linestyle=None, linewidth=2).set(title='Ferbuary 2024')
plt.xlabel('Days', fontsize=14)
plt.ylabel('Amount of Tips')
plt.show()







