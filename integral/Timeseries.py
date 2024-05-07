# numerical python(np)
# pandas +for data manipulation and analysis, data structure for plotting
#seaborn + data exploration and visualization
#matplotlib Data visualizatio and graphical representation
#scikit learn for linear regression

from datetime import date
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

Data1= pd.read_csv(r"C:\Users\pc\Desktop\Timeseries\Time Series Forecasting dataset\gold_monthly_csv.csv")
Data1 = pd.DataFrame(Data1)
print(Data1.head())
#       Date  Price
# 0  1950-01  34.73
# 1  1950-02  34.73
# 2  1950-03  34.73
# 3  1950-04  34.73
# 4  1950-05  34.73

print(Data1.shape) #847 rows, 2columns
print(Data1.isna().sum())
print(Data1.isnull().any().any()) #Returns a boolean statement

# kasaa = pd.read_csv(r"C:\Users\pc\Desktop\Timeseries\kasaa monthly totals.csv", index_col='Date')
# print(kasaa.head())

date = pd.date_range(start='1/1/1950', end='8/1/2020', freq='M')
print(date)

Data1['month'] = date
Data1.drop('Date', axis=1, inplace=True)
Data1 = Data1.set_index('month')
print(Data1.head())
#           
# Date          Records
# 1/1/2018       26
# 1/2/2018       28
# 1/3/2018       25
# 1/4/2018       40
# 1/5/2018       28

 # Checks and counts the null values.
# Date     0
# Price    0
# dtype: int64
# def timeset(Data1):
#             Data1['Date'] = pd.to_datetime(Data1['Date'])
#             Data1['month'] = Data1['Date'].dt.month
#             Data1.drop('Date', axis=1, inplace=True)
#             Data1.set_index('month', inplace=True)
#             Data1.head()
# timeset(Data1)
# def firstplot(Data1):
#     Data1.plot(figsize=(20, 8))
#     plt.title("Monthly gold prices")
#     plt.xlabel("Months")
#     plt.ylabel("Price")
#     plt.grid()
#     plt.show() # This line is necessary to display the plot
# firstplot(Data1)
# Data1 is your DataFrame
# if isinstance(Data1.index, pd.DatetimeIndex):
#     print("The index is a DatetimeIndex.")
# else:
#     print("The index is not a DatetimeIndex.")




def EDA(Data1):
    print(Data1.describe())

EDA(Data1)
#Results
#              Price
# count   847.000000
# mean    416.556906
# std     453.665313
# min      34.490000
# 25%      35.190000
# 50%     319.622000
# 75%     447.029000
# max    1840.807000
def boxplot1(Data1):
    Data1.index = pd.to_datetime(Data1.index)
    plt.figure(figsize=(30, 8))
    sns.boxplot(x=Data1.index.year, y=Data1.Price, data=Data1)
    plt.title("Monthly gold prices")
    plt.xlabel("Years")
    plt.ylabel("Price")
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()

boxplot1(Data1)


#KASARANI MODELLING

# def kasaaplot(kasaa):
#     kasaa.plot(figsize=(10,4))
#     plt.title("Crime in monthly")
#     plt.xlabel("months")
#     plt.ylabel("No of crimes")

#     plt.grid()
#     plt.show()

# kasaaplot(kasaa)




