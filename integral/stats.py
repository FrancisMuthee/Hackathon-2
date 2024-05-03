import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time

# A panda series is like a column in a table.

a=[1,6,6]
myvar=pd.Series(a)

print(a)
myvar = pd.Series(a, index=
                  ["x","y","z"])

print(myvar[1]) #Returns the second value.


calories = {'day1':420, "day2": 382, "day3":585}
var1 =pd.Series(calories)
print(var1)
print(calories)


#Dataframes are multi-dimensional tables.

data= {
    "calories":[488,125,698],
    "duration":[52,45,23]
}
# day1    420
# day2    382
# day3    585

var2 = pd.DataFrame(data)
print(var2)

#    calories  duration
# 0       488        52
# 1       125        45
# 2       698        23

print(var2.loc[1]) # locates a row

var2 = pd.DataFrame(data, index=['day1', 'day2', 'day3'])
print(var2)

#       calories  duration
# day1       488        52
# day2       125        45
# day3       698        23

data1 = pd.read_csv(r"C:\Users\pc\Desktop\RR\Iris.csv")
print(data1.head())

#    Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
# 0   1            5.1           3.5            1.4           0.2  Iris-setosa      
# 1   2            4.9           3.0            1.4           0.2  Iris-setosa      
# 2   3            4.7           3.2            1.3           0.2  Iris-setosa      
# 3   4            4.6           3.1            1.5           0.2  Iris-setosa      
# 4   5            5.0           3.6            1.4           0.2  Iris-setosa 

#Loading json
data3 = {
  "Duration":{
    "0":60,
    "1":60,
    "2":60,
    "3":45,
    "4":45,
    "5":60
  },
  "Pulse":{
    "0":110,
    "1":117,
    "2":103,
    "3":109,
    "4":117,
    "5":102
  },
  "Maxpulse":{
    "0":130,
    "1":145,
    "2":135,
    "3":175,
    "4":148,
    "5":127
  },
  "Calories":{
    "0":409,
    "1":479,
    "2":340,
    "3":282,
    "4":406,
    "5":300
  }
}
datajson = pd.DataFrame(data3)
print(datajson)

#    Duration  Pulse  Maxpulse  Calories
# 0        60    110       130       409
# 1        60    117       145       479
# 2        60    103       135       340
# 3        45    109       175       282
# 4        45    117       148       406
# 5        60    102       127       300

print(datajson.info())
datajson.to_csv('firstcsv.csv', index= False)
#index is set to false to remove the indexing

# <class 'pandas.core.frame.DataFrame'>
# Index: 6 entries, 0 to 5
# Data columns (total 4 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   Duration  6 non-null      int64
#  1   Pulse     6 non-null      int64
#  2   Maxpulse  6 non-null      int64
#  3   Calories  6 non-null      int64
# dtypes: int64(4)
# memory usage: 240.0+ bytes



#data4 = pd.read_csv(r"C:\Users\pc\Desktop\RR\pulse.csv")
#print(data4.head())

#new_df = df.dropna()
#df.dropna(inplace = True) remove all rows with null numbers.
#df.fillna(130, inplace = True) filling null value with a number.
#df["Calories"].fillna(130, inplace = True) replacing for a particular column

#Can replace null values with mean, median,mode
# x = df["calories"].mean()
# df["calories"].fillna(x, inplace = True)

#df.drop_duplicates(inplace = True) removes all duplicates
#print(df.duplicated()) checks duplicate
#df.corr()

#for loops
numbers = [14,20,30,40,60,51,110,112,150]

for x in numbers:
    print(x)


print('Nos greater than 100')
for y in numbers:
    if y > 100:
        print(y)

#Time is money
now = datetime.now()
print(now)
epochtime = int(time.time())
print(epochtime)

new_epoch = epochtime + (30*60)
print(new_epoch)

#A histogram is a graph used to represent the frequency distribution of a few data points of one variable
 

