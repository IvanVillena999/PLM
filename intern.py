from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np

# Create Decision Tree
features = ['raining']
X = [[0], [1]]
Y = [0, 1]

clf = tree.DecisionTreeClassifier()
clf.fit(X,Y)

# Visualize Tree
dotfile = open("dtree.dot", 'w')
tree.export_graphviz(clf, out_file = dotfile, feature_names=features, filled=True, rounded=True, impurity=False, class_names=['No Umbrella','Umbrella'])
dotfile.close()

print(clf.predict([[0]]))

