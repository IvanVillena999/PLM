import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy.stats
from scipy.stats import chi2

data = [[14, 14, 12], [41, 43, 41], [10, 15, 15], [1, 1, 1]]
print('\n', stats.chi2_contingency(data, correction=True))

data2 = [[39.66346154, 43.87019231, 41.46634615], [12.69230769, 14.03846154, 13.26923077], [0.95192308,  1.05288462,  0.99519231]]

rounded_data2 = [[39.66, 43.87, 41.47],
                 [12.69, 14.04, 13.27],
                 [0.95, 1.05, 1.00],
                 [0.00, 0.00, 0.00]]

print('The Chi-Square Test Statistics is: 1.19')
print('The p-values is: 0.977')
print('The Degree of Freedom is: 6')
print('Decision: The Null Hypothesis that the GENDER of the Endorser and the Decision to Purchase are independent of each other is ACCEPTED')

row_labels = ['Male', 'Female', 'LGBTQ+', 'Preferred not say']
column_labels = ['Male Endorser', 'Female Endorser', 'LGBTQ Endorser']
fig = plt.figure(figsize=(8,6))
sns.heatmap(rounded_data2, annot=True, cmap='Blues', fmt='.2f', cbar=False, yticklabels=row_labels, xticklabels=column_labels)
plt.title('Chi-Square Contigency Table')
plt.xlabel('Gender of Endorser')
plt.ylabel('Gender of Consumer')
plt.show()