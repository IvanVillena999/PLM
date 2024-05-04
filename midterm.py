import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('visitors.csv')
print(df.head(100))

summary1 = df['bill'].describe()
print(summary1)

print(df.groupby('gender')['label1'].describe().reset_index())
print(df.groupby('gender')['label2'].describe().reset_index())