import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr
import researchpy as rp
import numpy as np

df = pd.read_csv('timss.csv')
df.head()




df['belonging'].replace(999999, 0, inplace=True)
df['bullying'].replace(999999, 0, inplace=True)

df = df[(df['belonging'] != 0) & (df['bullying'] != 0)]

df = df.dropna()

print('The number of observed scores for Students Belonging scale:', len(df.belonging))
print('The number of observed scores for Students School Bullying Experience scale:', len(df.bullying))
print(df[['belonging', 'bullying']].head(5515))

summary1 = rp.summary_cont(df['belonging'])
summary2 = rp.summary_cont(df['bullying'])

print(f"Summary Measures for Students Belonging Scale:\n{summary1}")
print()
print(f"Summary Measures for Students Bullying Experience Scale:\n{summary2}")
print()

sns.regplot(data=df, x='bullying', y='belonging', ci=None)
r,p = pearsonr(df['bullying'], df['belonging'])
print('The pearson correlation coefficient is:', r)
if r == 1:
    print("Decision: There is a perfect positive relationship between the variables belonging and the bullying")
elif r < 1:
    print("Decision: There is a positive relationship between the variables belonging and the bullying")
elif r == 0:
    print("Decision: There is no relationship between the variables belonging and the bullying")
elif r < -1:
    print("Decision: There is negative relationship between the variables belonging and the bullying")


plt.annotate(f'r = {r:.2f}', xy=(0.7, 0.9), xycoords='axes fraction')
slope, intercept, r_value, p_value, std_err = linregress(df['bullying'], df['belonging'])
plt.plot(df['bullying'], slope * df['bullying'] + intercept, color='red', label='Regression Line')

print()
print("Regression Statistics:")
print('\nThe linear regression model is: Sense of Belonging Scores =', {intercept}, '+', {slope}, 'Bullying Experience Scores')
print('\nSlope:', slope)
print('\nIntercept:', intercept)
print('\nR-squared:', r_value**2)
print('\np-value:', p_value)
print('\nStandard Error of the Estimate:', std_err)


plt.xlabel('Students Belonging')
plt.ylabel('Students Bullying Experience')
plt.title('Scatter Plot with Regression Line')
plt.legend()
plt.grid(True)

plt.show()

