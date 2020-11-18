# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import copy
import seaborn as sns
import matplotlib.pyplot as plt

#Preprocessing

#Read dataset
#model,year,engineSize,transmission,mileage,fuelType,tax,mpg,engineSize
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

#print(data_train.head())
#print(data_train.info())

#check the column-wise distribution of null values:
#print(data_train.isnull().sum())

#print(data_train['model'].value_counts()) #count of each model
#print(data_train['model'].value_counts().count()) #distinct models

#PLOTTING OF MODELS
model_count = data_train['model'].value_counts()
sns.set(style="darkgrid")
sns.barplot(model_count.index, model_count.values, alpha=0.9)
plt.title('Frequency Distribution of Models')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Model', fontsize=12)
#plt.show()

#PIE CHART OF MODELS
labels = data_train['model'].astype('category').cat.categories.tolist()
counts = data_train['model'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
#plt.show()

#TURN INTO CATEGORICAL VALUES
data_train_cpy = data_train.copy()
data_test_cpy = data_test.copy()
data_train_cpy['model'] = data_train_cpy['model'].astype('category')
data_train_cpy['transmission'] = data_train_cpy['transmission'].astype('category')
data_train_cpy['fuelType'] = data_train_cpy['fuelType'].astype('category')
data_test_cpy['model'] = data_test_cpy['model'].astype('category')
data_test_cpy['transmission'] = data_test_cpy['transmission'].astype('category')
data_test_cpy['fuelType'] = data_test_cpy['fuelType'].astype('category')

#TRAIN: Label Encoding
data_train_cpy['model'] = data_train_cpy['model'].cat.codes
data_train_cpy['transmission'] = data_train_cpy['transmission'].cat.codes
data_train_cpy['fuelType'] = data_train_cpy['fuelType'].cat.codes
#TEST: Label Encoding
data_test_cpy['model'] = data_test_cpy['model'].cat.codes
data_test_cpy['transmission'] = data_test_cpy['transmission'].cat.codes
data_test_cpy['fuelType'] = data_test_cpy['fuelType'].cat.codes

print(data_train_cpy.head())

