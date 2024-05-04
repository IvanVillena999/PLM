import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt


df = pd.read_csv('intern3.csv')
print(df.head())


X = df[['age','company']]
y = df['apply']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


print("Accuracy of the Decision Tree:", accuracy_score(y_test, y_pred))


fig = plt.figure(figsize=(10,10))
_ = tree.plot_tree(clf,
                   feature_names=X.columns,
                   class_names=None,
                   filled=True)
plt.title('Decision Tree on Students who will apply in the company', fontsize=20)
plt.show()
