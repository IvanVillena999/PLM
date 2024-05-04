import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import researchpy as rp
import seaborn as sns

df = pd.read_csv('freddata.csv')


print(df.head(29))

print('\nThe number of observations is:', len(df))

summary1 = df.describe()
summary2 = rp.summary_cont(df['fredtips'])
summary3 = rp.summary_cont(df['fredhrs'])
print(f'Descriptive Statitics\n{summary1}')
print(f'Descriptive Statitics\n{summary2}')
print(f'Descriptive Statitics\n{summary3}')

#Generating a simple scatterplot
x = df['fredtips']
y = df['fredhrs']

plt.plot(x, y, 'o', color='blue', markersize=10, markerfacecolor='red', markeredgecolor='black', markeredgewidth=1)
plt.xlabel('Amount of Tips received by Fred')
plt.ylabel('Number of Working Hours')
plt.show()





