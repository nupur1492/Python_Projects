import pandas as pd
import scipy.stats
import ggplot
from ggplot import *
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import statsmodels

data = pd.read_csv("turnstile_data_master_with_weather.csv")
df_subway = pd.DataFrame(data)
print(df_subway.columns)

p1 = ggplot(aes('Hour', 'ENTRIESn_hourly', color = 'rain'), data = df_subway) + geom_point()
#print(p1)

#Null hypothesis: The hourly entries on rainy and non rainy days are the same.
# Use Welch's t-test
# P critical: 0.05

df_subway_rain = df_subway[df_subway.rain == 1]
df_subway_norain = df_subway[df_subway.rain == 0]

ttest = scipy.stats.ttest_ind(df_subway_rain.ENTRIESn_hourly, df_subway_norain.ENTRIESn_hourly, equal_var = False)

print("t-test:  ", ttest)

if ttest[1] < 0.05:
    print("Small p value, Null hypothesis rejected")

# Add day of the week to dataframe
df_subway.DATEn = pd.to_datetime(df_subway.DATEn)
df_subway['dayofweek'] = df_subway['DATEn'].apply(lambda x: x.strftime('%w'))


p1 = ggplot(aes(x = 'maxpressurei', y = 'ENTRIESn_hourly'), data = df_subway) + geom_point()
#print(p1)

p2 = ggplot(aes(x = 'maxtempi', y = 'ENTRIESn_hourly'), data = df_subway) + geom_point()
#print(p2)

p3 = ggplot(aes(x = 'meanwindspdi', y = 'ENTRIESn_hourly'), data = df_subway) + geom_point()
print(p3)

# p1, p2, p3 show normal distribution

# split dataframe into training and testing sets

train, test = cross_validation.train_test_split(df_subway, test_size=0.3)


#Regression:
import statsmodels.formula.api as sma
lm = sma.ols(formula = 'ENTRIESn_hourly ~ maxtempi + maxpressurei + meanwindspdi + Hour', data = train).fit()
print("Regression intercepts and coefficients", lm.params)

print(lm.summary(), lm.conf_int(), lm.pvalues)

# p < 0.05 will denote a relationship between features and output


test_columns = ['maxtempi', 'maxpressurei', 'meanwindspdi', 'Hour']
test = pd.DataFrame(test, columns=test_columns)
test.head()

#Predict using test set
pred = lm.predict(test)



# Specific features/input columns for training
feature_columns = ['Hour', 'maxpressurei', 'maxdewpti', 'maxtempi', 'meanwindspdi', 'rain', 'fog', 'precipi', 'thunder', 'dayofweek' ]

features = df_subway[feature_columns]
y = df_subway.ENTRIESn_hourly

from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(features, y)

print(reg.intercept_)
print(reg.coef_)

#calculate r squared
reg.score(features, y)


