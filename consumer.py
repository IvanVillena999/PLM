import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('consumer.csv', encoding='latin1')
print(df.head(35))

df['feedbacks'].fillna('None', inplace=True)
df['feedbacks'] = df['feedbacks'].astype(str)

print(df['feedbacks'].head(35))
print('Number of Feedbacks Received:', len(df['feedbacks']))
print('Number of No Responses (None): ', df['feedbacks'].apply(lambda x: x == 'None').sum())
print('Total Valid Responses:', len(df['feedbacks']) - df['feedbacks'].apply(lambda x: x == 'None').sum())

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
vectorizer = CountVectorizer()


x = vectorizer.fit_transform(df['feedbacks'])
print(x)

print(pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out()))

feature_rank = list(zip(vectorizer.get_feature_names_out(), [x[0] for x in x.sum(axis=0).T.tolist()]))

feature_rank = np.array(sorted(feature_rank, key=lambda x: x[1], reverse=True))

top6 = feature_rank[:6]

n = 10
plt.figure(figsize=(5,5))
plt.barh(-np.arange(n), feature_rank[:n, 1].astype(float), height=.6, align='center')
plt.yticks(ticks=-np.arange(n), labels=feature_rank[:n, 0], fontsize=7)
plt.xlabel('Most Frequent Feedbacks', fontsize=8)

plt.show()