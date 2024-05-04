import matplotlib.pyplot as plt
import pandas as pd

#Read the data
df = pd.read_csv('satisfy.csv')
print(df.head(30))

#Rating scale: 4-Excellent, 3-Very good, 2- Good, 1-Not good (Categorical data)

#Group the data
print(df.groupby('food')['label'].describe().reset_index())

#Create pie chart
y = ([10, 13, 10, 17])
food = ['Excellent', 'Very Good', 'Good', 'Not Good']
myexplode = [0, 0, 0, 0.2]
plt.pie(y, labels=food, autopct='%.0f%%', explode=myexplode)
plt.title('Food Satisfaction Ratings')
plt.show()