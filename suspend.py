import pandas as pd
import numpy as np
import researchpy as rp
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('suspend.csv')
print(df.head(20))

index = df['index']
weather = df['weather']
suspend = df['suspend']

print(rp.summary_cat(df['index']))
print(rp.summary_cat(df['weather']))
print(rp.summary_cat(df['suspend']))

likehood_yes = round((9/19)*(5/9)*(4/9), 2)
likehood_no = round((10/19)*(5/10)*(5/10), 2)

print('\nLikelihood of Suspending F2F:', likehood_yes)
print('Likelihood of Not Suspending F2F:', likehood_no)

print('\nProbability of Suspending F2F:', likehood_yes/(likehood_yes+likehood_no))
print('Probability of Not Suspending F2F:', likehood_no/(likehood_yes+likehood_no))

suspend_yes = likehood_yes/(likehood_yes+likehood_no)
not_suspend = likehood_no/(likehood_yes+likehood_no)

encoder = preprocessing.LabelEncoder()
heatEnc = encoder.fit_transform(index)
print(heatEnc)

weatherEnc = encoder.fit_transform(weather)
print(weatherEnc)

suspendEnc = encoder.fit_transform(suspend)
print(suspendEnc)

features = list(zip(heatEnc, weatherEnc))
print(features)

decision = ['Yes', 'Maybe', 'No']

suspendModel1 =KNeighborsClassifier(n_neighbors=5, metric='manhattan')

suspendModel1.fit(features,suspendEnc)

prediction1 = suspendModel1.predict([[0, 0]])
print('\nShould the President suspend the F2F? {0}'.format(decision[prediction1[0]]))

if not_suspend > suspend_yes:
    print('\nThe President will', not_suspend, 'probability NOT likely to Suspend F2F Classes')

elif suspend_yes > not_suspend:
    print('\nThe President will', suspend_yes, 'probability LIKELY to Suspend F2F Classes')




