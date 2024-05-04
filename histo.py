import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import researchpy as rp
import seaborn as sns

df = pd.read_csv('freddata.csv')

print(df.tail(30))

sns.barplot(df, x='days', y='cust', estimator='sum', errorbar=None, color='green')
plt.xlabel('Month of February')
plt.ylabel('Number of Customers')
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


sns.histplot(data=df, x='diff', color='red')
plt.xlabel('Difference in Number of Customers')
plt.show()







