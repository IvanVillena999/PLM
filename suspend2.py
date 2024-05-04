import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing

df = pd.read_csv('suspend.csv')
print(df.head(20))

heat = df['heat']
weather = df['weather']
suspend = df['suspend']

encoder = preprocessing.LabelEncoder()
heatEnc = encoder.fit_transform(heat)
print(heatEnc)

weatherEnc = encoder.fit_transform(weather)
print(weatherEnc)

suspendEnc = encoder.fit_transform(suspend)
print(suspendEnc)

