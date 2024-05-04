import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import researchpy as rp

df = pd.read_csv("freddata.csv")
print(df.head(10))

y = df['fredtips']

#Bargraph seaborn
sns.barplot(df, x='days', y='fredtips', estimator='sum', errorbar=None, color='green')
plt.xlabel('February')
plt.ylabel('Amount of Tips Received by Fred')
plt.show()

#boxplot matplotlib
box = df['fredtips']
fig = plt.figure(figsize=(10, 7))
plt.boxplot(box)
plt.xlabel('Amount of Tips Received by Fred')
plt.show()

#Boxplot seaborn
sns.boxplot(x='fredtips', data=df, color='red')
plt.xlabel('Amount of Tips Received by Fred')
plt.show()

#Line graph matplotib
x = df['days']
y = df['fredtips']

plt.plot(x, y, color='blue')
plt.xlabel('February', fontsize=10)
plt.ylabel('Amount of Tips')
plt.show()

#Line graph seaborn
sns.lineplot(x='days', y='fredtips', data=df, color='magenta', linestyle='dotted', linewidth=5).set(title='Tips of Fred')
plt.xlabel('February')
plt.ylabel('Amount of Tips Received by Fred')
sns.set_style(style='white')
plt.show()



