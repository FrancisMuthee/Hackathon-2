import pandas as pd
from datetime import date
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats
import pylab
from autots import AutoTS

kasaa1 = pd.read_csv(r"C:\Users\pc\Desktop\Timeseries\kasaa monthly totals.csv")
kasa2 = pd.DataFrame(kasaa1)
print(kasa2.head())

date = pd.date_range(start='1/1/2018', end='12/12/2022', freq='MS')
kasa2['Date'] = date
kasa2 = kasa2.assign(month=date).drop('Date', axis=1).set_index('month')
print(kasa2.head())
print(kasa2.isna().sum())


# kasa2.plot()
# plt.show()


# print(kasa2.describe())
# bins = [10,20,30,40,50,60,70,80,90]
# plt.hist(kasa2.Records, bins=bins, edgecolor='black')
# plt.show()

#QQ plot
# fit = scipy.stats.probplot(kasa2.Records, plot=pylab) 
# pylab.show()

result = adfuller(kasa2['Records'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
        print(f'   {key}: {value}')

# Extract time series data
# ts_data = kasa2['Records']
# ts_data.plot()
# plt.show()

# best_aic = float('inf')
# best_order = None
# best_model = None

# for p in range(5):
#     for d in range(3):
#         for q in range(5):
#             try:
#                 model = ARIMA(ts_data, order=(p, d, q))
#                 model_fit = model.fit()
#                 if model_fit.aic < best_aic:
#                     best_aic = model_fit.aic
#                     best_order = (p, d, q)
#                     best_model = model_fit
#             except:
#                 continue

# print(f'Best ARIMA{best_order} AIC: {best_aic}')

# Best ARIMA(1, 1, 1) AIC: 469.01670919539185


# Fit ARIMA model
# model = ARIMA(ts_data)
# fitted_model = model.fit()
# print(fitted_model.summary())

# residuals =fitted_model.resid
# residuals.plot()
# plt.show()


#                                SARIMAX Results
# ==============================================================================
# Dep. Variable:                Records   No. Observations:                   60
# Model:                 ARIMA(4, 2, 4)   Log Likelihood                -230.215
# Date:                Wed, 08 May 2024   AIC                            478.429
# Time:                        15:10:04   BIC                            496.973
# Sample:                    01-01-2018   HQIC                           485.652
#                          - 12-01-2022
# Covariance Type:                  opg
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# ar.L1         -1.2497     27.907     -0.045      0.964     -55.947      53.448
# ar.L2         -1.2312      6.845     -0.180      0.857     -14.646      12.184
# ar.L3         -1.2865     27.403     -0.047      0.963     -54.996      52.423
# ar.L4         -0.3051      8.419     -0.036      0.971     -16.806      16.196
# ma.L1         -0.0744    592.182     -0.000      1.000   -1160.729    1160.580
# ma.L2         -0.0120     72.147     -0.000      1.000    -141.418     141.394
# ma.L3          0.0744    587.246      0.000      1.000   -1150.907    1151.055
# ma.L4         -0.9880     27.661     -0.036      0.972     -55.202      53.226
# sigma2       143.7259      7.612     18.881      0.000     128.806     158.646
# ===================================================================================
# Ljung-Box (L1) (Q):                   0.10   Jarque-Bera (JB):                 2.06
# Prob(Q):                              0.75   Prob(JB):                         0.36
# Heteroskedasticity (H):               2.53   Skew:                            -0.45
# Prob(H) (two-sided):                  0.05   Kurtosis:                         3.17
# ===================================================================================


# ADF Statistic: -3.546552929508487
# p-value: 0.006868978424718734
# Critical Values:
#    1%: -3.5463945337644063
#    5%: -2.911939409384601
#    10%: -2.5936515282964665


# add_decompose = seasonal_decompose(kasa2['Records'], model='additive', extrapolate_trend='freq')

# add_decompose.plot()
# plt.show()

# model = AutoTS(forecast_length=10, frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
# model = AutoTS(forecast_length=10)
# model_fit = model.fit(kasa2)
# print(model_fit.summary())

# Forecast future time periods
# forecast_steps = 12  # Change this according to the number of steps you want to forecast
# forecast = model_fit.forecast(steps=forecast_steps)

# # Print forecasted values
# print("Forecasted Values:")
# print(forecast)
# forecast.plot()
# plt.show()



# Plot original time series data and forecasted values
# plt.plot(ts_data, label='Original Data')
# plt.plot(range(len(ts_data), len(ts_data) + forecast_steps), forecast, label='Forecast')
# plt.title('Original Data and Forecast')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
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






