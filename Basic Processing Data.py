#Import Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import seaborn as sns
# Read CSV file
df = pd.read_csv("salaries.csv")
#List first 5 records
df.head()
#Check a particular column type
df['salary'].dtype
#Check types for all the columns
df.dtypes

#Group data using rank
df_rank = df.groupby(['rank'])

#Calculate mean value for each numeric column per each group
df_rank.mean()
#Calculate mean salary for each professor rank:
df.groupby('rank')[['salary']].mean()

#Calculate mean salary for each professor rank:
df.groupby(['rank'], sort=False)[['salary']].mean()

#Calculate mean salary for each professor rank:
df_sub = df[ df['salary'] > 120000 ]

#Select only those rows that contain female professors:
df_f = df[ df['sex'] == 'Female' ]

# DATA FRAME SLICING
#Select column salary:

df['salary']
#Select column salary:
df[['rank','salary']]
#Select rows by their position:
df[10:20]
#Select rows by their labels:
df_sub.loc[10:20,['rank','sex','salary']]
#Select rows by their labels:
df_sub.iloc[10:20,[0, 3, 4, 5]]

df.iloc[0]  # First row of a data frame
#df.iloc[i]  #(i+1)th row 
df.iloc[-1] # Last row 
df.iloc[:, 0]  # First column
df.iloc[:, -1] # Last column 
df.iloc[0:7]       #First 7 rows 
df.iloc[:, 0:2]    #First 2 columns
df.iloc[1:3, 0:2]  #Second through third rows and first 2 columns
df.iloc[[0,5], [1,3]]  #1st and 6th rows and 2nd and 4th columns

# SORTING
# Create a new data frame from the original sorted by the column Salary
df_sorted = df.sort_values( by ='service')
df_sorted.head()

# We can sort the data using 2 or more columns:
df_sorted = df.sort_values( by =['service', 'salary'], ascending = [True, False])
df_sorted.head(10)

# MISSING VALUES
# Read a dataset with missing values
flights = pd.read_csv("flights.csv")
# Select the rows that have at least one missing value
flights[flights.isnull().any(axis=1)].head()
flights[['dep_delay','arr_delay']].agg(['min','mean','max']) #agg() method are useful when multiple statistics are computed per column:


# LINE PLOT
x = flights['dep_time']
y = flights['arr_time']
plt.scatter(x, y)


# BASIC DESCRIPTIVE STATISTIC
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import seaborn as sns

flights = pd.read_csv("flights.csv")
flights.describe()
x1 = flights.groupby(['arr_time'])
x1.describe
x2 = flights['arr_time']
x2.describe()
x2 = flights['arr_time']
x2.mean()
x2.std()

