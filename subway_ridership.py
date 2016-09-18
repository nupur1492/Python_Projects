import pandas as pd
import scipy.stats
import ggplot
from ggplot import *
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


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



if ttest[1] > 0.05:
    print("Significant p value, Null hypothesis rejected")

# Add day of the week to dataframe
df_subway.DATEn = pd.to_datetime(df_subway.DATEn)
df_subway['dayofweek'] = df_subway['DATEn'].apply(lambda x: x.strftime('%w'))


# split dataframe into training and testing sets

train, test = cross_validation.train_test_split(df_subway, test_size=0.3)

# Specific features/input columns for training
feature_columns = ['Hour', 'maxpressurei', 'maxdewpti', 'maxtempi', 'meanwindspdi', 'rain', 'fog', 'precipi', 'thunder', 'dayofweek' ]

features_train = pd.DataFrame(train, columns=feature_columns)
#output column
labels_train = train['ENTRIESn_hourly']

# divide test into input and output
features_test = pd.DataFrame(test, columns=feature_columns)
labels_test = np.array(test['ENTRIESn_hourly'])
#labels_test = test['ENTRIESn_hourly']

print(len(features_train), len(labels_train), len(features_test), len(labels_test))


#Machine Learning
# 1. Multiple Linear Regression
from sklearn import linear_model

reg_mlr = linear_model.LinearRegression()
reg_mlr.fit(features_train,labels_train)

reg_pred = reg_mlr.predict(features_test)
print(reg_pred[1:10])
print(labels_test[1:10])

#print('Multiple Linear Regression Score:  ', reg_mlr.score(features_test, labels_test))
#print('Multiple Linear Regression Score:  ', reg_mlr.score(labels_test, reg_pred))
print('Residual Sum of Squares: ', np.mean(abs(reg_pred - labels_test)))

# Plot Regression:
plt.figure()
#plt.scatter(features_test, labels_test, color = 'black')
plt.plot(features_test, reg_pred, color = 'blue')

plt.show()


# Decision Trees
from sklearn import tree

reg_dt = tree.DecisionTreeRegressor(max_depth=5)

reg_dt.fit(features_train, labels_train)

pred_dt = reg_dt.predict(features_test)

#print(type(pred_dt), pred_dt)
#print(type(labels_test), labels_test)

#print('Decision Tree Regression Score: ', reg_dt.score(labels_test, pred_dt))
