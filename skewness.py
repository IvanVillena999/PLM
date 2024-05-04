import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import scipy
from scipy.stats import skew, kurtosis

#Read the dataset file
df = pd.read_csv('overtime.csv')
print(df.head(70))

#Summary measures (central tendency)
print(df.describe().round(decimals=0))

#Generate Histogram
fig, ax = plt.subplots(figsize=(6, 4))
df['fred'].plot(kind='hist')
skew_fred = df['fred'].skew() #Compute for skewness
kurt_fred = df['fred'].kurt() #Compute for kurtosis
ax.text(0.7, 0.9, f'Skewness: {skew_fred:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')
ax.text(0.7, 0.9, f'Kurtosis: {kurt_fred:.2f}', transform=ax.transAxes, fontsize=10, verticalalignment='bottom')
plt.show()

#Generate Kernel Distribution Estimates (KDE) to determine the places of mean, median mode
sns.kdeplot(df['fred'], color='red')
sns.despine(top=True, right=True, left=True)
plt.xticks([])
plt.yticks([])
plt.xlabel('')
plt.ylabel('')
plt.title('KDE Plot of Fred Overtime: Skewness and Kurtosis')

#Show the Central tendencies
plt.axvline(df['fred'].mean(), color='black', label='Mean')
plt.axvline(df['fred'].median(), color='green', label='Median')
plt.axvline(df['fred'].mode().squeeze(), color='yellow', label='Mode')
plt.legend()
plt.show()

#Plotly histogram
dist_fred = df['fred']
fig_dist = ff.create_distplot([dist_fred], ['fred'])
fig_dist.show()