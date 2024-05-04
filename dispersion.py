import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import researchpy as rp

df = pd.read_csv('tips.csv')
print(df.head(20))

def compute_dispersion(df):
    variance1 = df['fredtips'].var()
    variance2 = df['berttips'].var()
    std1 = df['fredtips'].std()
    std2 = df['berttips'].std()
    print('\nThe variance for Fredtips is:', variance1)
    print('\nThe variance for Berttips is:', variance2)
    print('\nThe standard deviation for Fredtips is:', std1)
    print('\nThe standard deviation for Fredtips is:', std2)

dispersion = input('Do you want to know the dispersion? (y/n)')
if dispersion == 'y':
    print(compute_dispersion(df))
if dispersion == 'n':
    print(df.describe())




