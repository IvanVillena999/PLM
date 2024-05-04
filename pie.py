import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Read the dataset (satisfy.csv)
df = pd.read_csv('satisfy.csv')
print(df.head(20))

def pie_chart(df):
    #Rating scale: 4-Excellent, 3-Very Good, 2-Good, 1-Not Good
    print(df.groupby('food')['label'].describe().reset_index())

    #Create the pie chart. Sort the values/count from highest to lowest
    y = [10, 13, 10, 17]
    food = ['Excellent', 'Very Good', 'Good', 'Not Good']
    plt.pie(y, labels=food, autopct='%.0f%%')
    plt.title("Food Satisfaction Ratings for the Month of March")
    plt.show()


    print(df.groupby('service')['label2'].describe().reset_index())

    service_ratings = [20, 15, 10, 5]
    myexplode2 = [0, 0.2, 0, 0]
    service = ['Excellent', 'Very Good', 'Good', 'Not Good']
    plt.pie(service_ratings, labels=service, autopct='%.0f%%', explode=myexplode2)
    plt.title("Service Satisfaction Ratings for the Month of March")
    plt.show()

pie = input('Do you want to show the pie graphs (y/n)')
if pie == 'y':
    print(pie_chart(df))
if pie == 'n':
    print('None')









