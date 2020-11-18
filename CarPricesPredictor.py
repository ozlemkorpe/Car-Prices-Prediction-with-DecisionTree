#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy import stats
from scipy.stats import norm, skew

#Preprocessing

#Read dataset
#model,year,price,transmission,mileage,fuelType,tax,mpg,engineSize
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

#print(data_train)

#data_train.head().T
#plt.plot(data_train["model"])
#plt.boxplot(data_train["engineSize"])
#plt.show()

data_train["price"] =np.log1p(data_train["price"])

#Scatter plot
plt.figure(figsize=[8,6])
plt.scatter(x=data_train['model'], y=data_train['price'])
plt.xlabel('model', fontsize=12)
plt.ylabel('price', fontsize=12)
#plt.show()

#Features
ntrain = data_train.shape[0]
ntest = data_test.shape[0]
features_train = data_train.drop(["price"], axis = 1)
y_train = data_train["price"]
data_features = pd.concat((features_train, data_test)).reset_index(drop = True)

#Number of categorical columns
cols2 = data_features.columns
cols = data_features.select_dtypes([np.number]).columns
str_ = set(cols2)-set(cols)


#Missing Data
data_features.isnull().sum()[data_features.isnull().sum() > 0].sort_values(ascending = False)
#data percent
total_missed = data_features.isnull().sum().sort_values(ascending=False)
percent = (data_features.isnull().sum()/data_features.isnull().count()).sort_values(ascending=False)
data_missed = pd.concat([total_missed, percent], axis=1, keys=['Total Missed', 'Percent'])
data_missed.head(10)

train = data_features[:ntrain]
test = data_features[ntrain:]
X_train = train
