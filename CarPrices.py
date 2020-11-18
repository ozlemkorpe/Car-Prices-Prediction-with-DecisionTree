# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image
from sklearn import tree
import pydotplus

#PREPROCESSING
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
model_count = data_train['transmission'].value_counts()
sns.set(style="darkgrid")
sns.barplot(model_count.index, model_count.values, alpha=0.9)
plt.title('Frequency Distribution of transmission')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('transmission', fontsize=12)
plt.show()

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

#print(data_train_cpy.head())
#print(data_train_cpy.info())

#split dataset in features and target variable
feature_cols = ['model', 'year', 'engineSize', 'transmission','mileage','fuelType','mpg' ,'engineSize']
X = data_train_cpy[feature_cols] # Features
y = data_train_cpy.price # Target variable

Xt = data_test_cpy[feature_cols] # Features
yt = data_test_cpy.price # Target variable

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=100)

# Train Decision Tree Classifer
clf = clf.fit(X,y)

#Predict the response for test dataset
y_pred = clf.predict(Xt)

# Model Accuracy: Is not suitable for continous values
print("Accuracy:",metrics.accuracy_score(yt, y_pred))
# Model Score: Is suitable for continous values, how close the predicted values to actual values
print("Score:", r2_score(yt, y_pred))

#Other option for splitting one dataset into two to test accuracy
"""
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=48)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
"""

#Print tree on console
text_representation = tree.export_text(clf)
#print(text_representation)

"""
# Create DOT data
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=data_test_cpy. ,class_names=yt)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
Image(graph.create_png())
"""