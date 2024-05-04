import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
import scipy
from scipy.stats import skew, kurtosis

df = pd.read_csv('overtime.csv')
print(df.head(20))

print(df.describe().round(decimals=0))


fig, ax = plt.subplots(figsize=(6, 4))
df['fred'].plot(kind='hist')
skew_fred = df['fred'].skew()
kurt_fred = df['fred'].kurtosis()
ax.text(0.7, 0.9, f'Skewness: {skew_fred:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
ax.text(0.7, 0.9, f'Kurtosis: {kurt_fred:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='bottom')
plt.show()

sns.kdeplot(df['fred'], color='red')
sns.despine(top=True, right=True, left=True)
plt.xticks([])
plt.yticks([])
plt.xlabel('')
plt.ylabel('')
plt.title('KDE Plot for Fred Overtime: Skewness vs Kurtosis')

plt.axvline(df['fred'].mean(), color='black', label="Mean: 4")
plt.axvline(df['fred'].median(), color='green', label="Median: 4")
plt.axvline(df['fred'].mode().squeeze(), color='yellow', label="Mode: 1")
plt.legend()
plt.show()


dist_fred = df['fred']
fig_dist = ff.create_distplot([dist_fred], ['fred'])
fig_dist.show()