import pandas as pd
import numpy as np
from datetime import date
import scipy
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm
import pylab
from autots import AutoTS
from IPython.display import Image

# Loading the data
kasaa1 = pd.read_csv(r"C:\Users\pc\Desktop\Timeseries\kasaa monthly totals.csv")

# Convert the data to a dataframe and display the firsy five observations.
kasa2 = pd.DataFrame(kasaa1)
print(kasa2.head())

# change the date format in a monthly frequency, and setting the colum date to month
date = pd.date_range(start='1/1/2018', end='12/12/2022', freq='MS')
# kasa2['Date'] = date
kasa2 = kasa2.assign(month=date).drop('Date', axis=1).set_index('month')
print(kasa2.head())
#checking the no. of null values
print(kasa2.isna().sum())


# kasa2.plot()
# plt.show()

#Performing a descriptive statistics
print(kasa2.describe())

#Plotting a histogram of the distribution.

# bins = [10,20,30,40,50,60,70,80,90]
# plt.hist(kasa2.Records, bins=bins, edgecolor='black')
# plt.title('Histogram for the records of crime')
# plt.xlabel('year/months')
# plt.ylabel("counts")
# plt.show()

# print(kasa2.Records.corr())

#QQ plot
#To determine the distribution of the sample. The points should follow a normal distribution.
#Or a y=x
# fit = scipy.stats.probplot(kasa2.Records, plot=pylab) 
# plt.savefig("qq.png")
# pylab.show()


#Unit-root test
# Unit root is a characteristic of a time series that makes it non-stationary.Its presence signifies non-stationarity.
#The ADF , is used to check staionarity. if the 
# result = adfuller(kasa2['Records'])
# print('ADF Statistic:', result[0])
# print('p-value:', result[1])
# print('Critical Values:')
# for key, value in result[4].items():
#         print(f'   {key}: {value}')

        # ADF Statistic: -3.546552929508487
        # p-value: 0.006868978424718734
        # Critical Values:
        # 1%: -3.5463945337644063
        # 5%: -2.911939409384601
        # 10%: -2.5936515282964665
#The p-value is less than 0.05 therefore we rejext the null hypothesis and hence the series is stationary.


#Extract time series data
ts_data = kasa2['Records']
ts_data_diff = ts_data.diff().dropna()


# ts_data_diff.plot()
# plt.save.fig("diff1.png")
# plt.show()

result = adfuller(ts_data_diff)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
        print(f'   {key}: {value}')

# ADF Statistic: -6.799428071189134
# p-value: 2.2568585308344013e-09
# Critical Values:
#    1%: -3.5506699942762414
#    5%: -2.913766394626147
#    10%: -2.5946240473991997

# The more negative the ADF statistic, the stronger the evidence against the presence of a unit root, indicating a more stationary series.


best_aic = float('inf')
best_order = None
best_model = None

for p in range(5):
    for d in range(3):
        for q in range(5):
            try:
                model = ARIMA(ts_data_diff, order=(p, d, q))
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = (p, d, q)
                    best_model = model_fit
            except:
                continue

print(f'Best ARIMA{best_order} AIC: {best_aic}')
# Best ARIMA(1, 0, 1) AIC: 467.4196792736337
coefficients = model_fit.params
print(coefficients)



# Best ARIMA(1, 0, 1) AIC: 467.4196792736337
# ar.L1      -2.093387
# ar.L2      -1.601968
# ar.L3      -0.584263
# ar.L4      -0.138058
# ma.L1      -0.113537
# ma.L2      -1.757105
# ma.L3      -0.099164
# ma.L4       0.989880
# sigma2    150.750160

# Invert differencing to get back to the original scale
# forecasted_values = model_fit.forecast(steps=len(ts_data_diff))
# forecasted_values = np.cumsum(forecasted_values)  # Cumulative sum to invert differencing

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(ts_data.index, ts_data.values, label='Actual', marker='o')
# plt.plot(ts_data_diff.index, ts_data_diff.values, label='Differenced Actual', marker='o')
# plt.plot(forecasted_values.index, forecasted_values, label='Forecasted', marker='o')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.title('Actual vs. Forecasted Values')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("All_in_one.png")
# plt.show()


# Forecast future time periods
# forecast_steps = 10  # Change this according to the number of steps you want to forecast
# forecast = model_fit.forecast(steps=forecast_steps)
# # # Print forecasted values
# print("Forecasted Values:")
# print(forecast)
# forecast.plot()
# plt.savefig("forecast.png")
# plt.show()

# Forecasted Values:
# 2023-01-01    12.130980
# 2023-02-01     1.714990
# 2023-03-01     0.999115
# 2023-04-01    -0.924260
# 2023-05-01     5.201934
# 2023-06-01    -2.415409
# 2023-07-01     5.208957
# 2023-08-01    -1.593135
# 2023-09-01     4.306745
# 2023-10-01    -0.280563

# add_decompose = seasonal_decompose(
# ts_data_diff, 
# model='additive', 
# extrapolate_trend='freq'
#  )
# residuals = add_decompose.resid
# plot_pacf(residuals, lags=20, zero=False)
# plt.show()
# add_decompose.plot()
# plt.show()
# fit = scipy.stats.probplot(ts_data_diff, plot=pylab) 
# plt.savefig("qq3.png")
# pylab.show()
# residuals = add_decompose.resid
# residuals = add_decompose.resid
# print(residuals)
# residuals.plot()
# plt.show()


# Fit ARIMA model
# model = ARIMA(ts_data_diff)
# fitted_model = model.fit()
# print(fitted_model.summary())

                            #  SARIMAX Results
# ==============================================================================
# Dep. Variable:                Records   No. Observations:                   59
# Model:                          ARIMA   Log Likelihood                -236.642
# Date:                Thu, 09 May 2024   AIC                            477.285
# Time:                        12:48:13   BIC                            481.440
# Sample:                    02-01-2018   HQIC                           478.907
#                          - 12-01-2022
# Covariance Type:                  opg
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          0.4068      1.830      0.222      0.824      -3.180       3.993
# sigma2       178.3887     33.845      5.271      0.000     112.053     244.724
# ===================================================================================
# Ljung-Box (L1) (Q):                   4.56   Jarque-Bera (JB):                 2.01
# Prob(Q):                              0.03   Prob(JB):                         0.37
# Heteroskedasticity (H):               2.09   Skew:                            -0.45
# Prob(H) (two-sided):                  0.11   Kurtosis:                         3.09
# ===================================================================================






# plt.hist(residuals, bins=30, color='skyblue', edgecolor='black')
# plt.show()

# Forecast future time periods
# forecast_steps = 12  # Change this according to the number of steps you want to forecast
# forecast = model_fit.forecast(steps=forecast_steps)

# # Print forecasted values
# print("Forecasted Values:")
# print(forecast)
# forecast.plot()
# plt.show()





Before_covid = kasa2['2018-01-01':'2019-12-12']
sum_total = Before_covid['Records'].sum()

# Before_covid.boxplot(column='Records')
# plt.show()

print("Total sum of values for the time before covid-19:",  sum_total)
#Total sum of values for the time before covid-19: 868

After_covid = kasa2['2020-01-01':'2022-12-12']
sum_total2= After_covid['Records'].sum()

After_covid.boxplot(column='Records')
plt.show()

print("Total sum of values for the time after covid-19:",  sum_total2)
#Total sum of values for the time after covid-19: 1609






