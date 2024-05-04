import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('lotto.csv')
print(df.head(20))

df['win'] = df['win'].astype(str)

print('\nThe number of winning combinations:', len(df['win']))

#Bag of Words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
vectorizer = CountVectorizer()


x = vectorizer.fit_transform(df['win'])
print(x)

print(pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out()))

feature_rank = list(zip(vectorizer.get_feature_names_out(), [x[0] for x in x.sum(axis=0).T.tolist()]))

feature_rank = np.array(sorted(feature_rank, key=lambda x: x[1], reverse=True))

top6 = feature_rank[:6]

n = 7
plt.figure(figsize=(5,5))
plt.barh(-np.arange(n), feature_rank[:n, 1].astype(float), height=.6, align='center')
plt.yticks(ticks=-np.arange(n), labels=feature_rank[:n, 0], fontsize=12)
plt.xlabel('Most Frequent Winning Numbers (Jan to Mar 2024)', fontsize=8)

plt.show()

lp = input('Do you want a lucky pick numbers? (y/n)')
if lp == 'y':
    print(f'Here are your lucky numbers: \n{top6}')
if lp == 'n':
    print('I will pick my own numbers')







