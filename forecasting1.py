# -*- coding: utf-8 -*-
"""

@author: Gopinath
"""


#Loading dataset
import pandas as pd
import numpy as np
df=pd.read_excel("Airlines+Data.xlsx",header=0,parse_dates=True)
df
df.info()
df.describe()

#EDA
df.set_index('Month',inplace=True)
df
df.isnull().sum().sum()
df[df.duplicated()].shape
df[df.duplicated()]
df.drop_duplicates(inplace=True)
df

#visualizations
from matplotlib import pyplot
import seaborn as sns
#density plot
df.plot(kind='kde')
pyplot.show()
#box and whisker plot
df.boxplot()
pyplot.show()
#histogram
df.hist()
pyplot.show()
#lag plot
from pandas.plotting import lag_plot
lag_plot(df,lag=1)
pyplot.show()
#autocorrelation

from statsmodels.graphics.tsaplots import plot_acf
pyplot.figure(figsize=(40,10))
plot_acf(df,lags=20)
pyplot.grid()
pyplot.show()

#sampling
upsampled=df.resample('M').mean()
print(upsampled.head(32))

#interpolate the missing value
interpolated = upsampled.interpolate(method='linear')
print(interpolated.head(15))
interpolated.plot()
interpolated.shape
df.plot()
#downsampling

resample=df.resample('Q')
quarterly_mean=resample.mean()
quarterly_mean.plot()
quarterly_mean.shape
#transformations
#lineplot
pyplot.subplot(211)
pyplot.plot(df)

#histogram
pyplot.subplot(212)
pyplot.hist(df)
pyplot.show()

#square root transform
from numpy import sqrt
from pandas import DataFrame
data=DataFrame(df.values)
data.columns=['Passengers']
data['Passengers']=sqrt(data['Passengers'])
data.min()
data.max()

#lineplot
pyplot.subplot(211)
pyplot.plot(df)
#histogram
pyplot.subplot(212)
pyplot.hist(df)
pyplot.show()

#log transform
from numpy import log
data=DataFrame(df.values)
data.columns=['Passengers']
data['Passengers']=log(data['Passengers'])
data.min()
data.max()
print(quarterly_mean.head())
quarterly_mean.plot()
#lineplot
pyplot.subplot(211)
pyplot.plot(data)
#histogram
pyplot.subplot(212)
pyplot.hist(data)
pyplot.show()

#forecasting methods
interpolated
interpolated.reset_index(inplace=True)
interpolated['t']=1
for i,row in interpolated.iterrows():
  interpolated['t'].iloc[i] = i+1
interpolated
interpolated['t_sq'] = (interpolated['t'])**2
interpolated["month"] = interpolated.Month.dt.strftime("%b") # month extraction
interpolated["year"] = interpolated.Month.dt.strftime("%Y") # year extraction
interpolated
months=pd.get_dummies(interpolated["month"])
months=months[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]
months
df=pd.concat([interpolated,months],axis=1)
df['log_Passengers'] = np.log(df['Passengers'])
df
pyplot.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=df,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

pyplot.figure(figsize=(12,3))
sns.lineplot(x="year",y="Passengers",data=df)

#splitting of data
train = df.head(81) # training data
test = df.tail(14) # test Data

#model based
#1.linear
import statsmodels.formula.api as smf
linear=smf.ols('Passengers~t',data=train).fit()
pred_linear=pd.Series(linear.predict(pd.DataFrame(test['t'])))
rmse_linear=np.sqrt(np.mean((np.array(test['Passengers']-np.array(pred_linear))**2)))
rmse_linear
test['t']
#2.exponential
exp=smf.ols('log_Passengers~t',data=train).fit()
pred_exp=pd.Series(exp.predict(pd.DataFrame(test['t'])))
rmse_exp=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_exp)))**2))
rmse_exp
test['t']
#3.quadratic
quad = smf.ols('Passengers~t+t_sq',data=train).fit()
pred_quad = pd.Series(quad.predict(test[["t","t_sq"]]))
rmse_quad = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_quad))**2))
rmse_quad
#4.additive seasonality
add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea = pd.Series(add_sea.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea
#5.Additive Seasonality Quadratic
add_sea_Quad = smf.ols('Passengers~t+t_sq+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_sq']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad
#5.Multiplicative Seasonality
Mul_sea = smf.ols('log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

#6.Multiplicative Additive Seasonality
Mul_Add_sea = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea
#Comparing the results
data = {"MODEL":pd.Series(["rmse_linear","rmse_exp","rmse_quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_exp,rmse_quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])

#data based
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#train and test
#moving average
pyplot.figure(figsize=(12,4))
interpolated.Passengers.plot(label="org")
for i in range(2,24,6):
    interpolated["Passengers"].rolling(i).mean().plot(label=str(i))
pyplot.legend(loc='best')

#time series decomposition
decompose_ts_add = seasonal_decompose(interpolated.Passengers,period=12)
decompose_ts_add.plot()
pyplot.show()

#acf &pacf
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(interpolated.Passengers,lags=14)
tsa_plots.plot_pacf(interpolated.Passengers,lags=14)
pyplot.show()

#evaluating meric MAPE
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)
#1.simple exponential method
ses_model = SimpleExpSmoothing(train["Passengers"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = test.index[0],end = test.index[-1])
MAPE(pred_ses,test.Passengers)
#2.holt method
hw = Holt(train["Passengers"]).fit(smoothing_level=0.1, smoothing_slope=0.2)
pred_hw = hw.predict(start = test.index[0],end = test.index[-1])
MAPE(pred_hw,test.Passengers)

#Holts winter exponential smoothing with additive seasonality and additive trend
hwe_add_add = ExponentialSmoothing(train["Passengers"],seasonal="add",trend="add",seasonal_periods=12).fit(smoothing_level=0.1, smoothing_slope=0.2) #add the trend to the model
pred_hwe_add_add = hwe_add_add.predict(start = test.index[0],end = test.index[-1])
MAPE(pred_hwe_add_add,test.Passengers)

#Holts winter exponential smoothing with multiplicative seasonality and additive trend

from sklearn.metrics import mean_squared_error
from numpy import sqrt
hw_mul_add = ExponentialSmoothing(train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit(smoothing_level=0.1, smoothing_slope=0.2)
pred_hw_mul_add = hw_mul_add.predict(start = test.index[0],end = test.index[-1])
MAPE(pred_hw_mul_add,test.Passengers)
rmse_hw_mul_add = sqrt(mean_squared_error(pred_hw_mul_add,test.Passengers))
rmse_hw_mul_add
#Final Model by combining train and test
hw_model_add_add = ExponentialSmoothing(interpolated["Passengers"],seasonal="add",trend="add",seasonal_periods=10).fit()
#Forecasting for next 10 time periods
hw_model_add_add.forecast(10)






































































