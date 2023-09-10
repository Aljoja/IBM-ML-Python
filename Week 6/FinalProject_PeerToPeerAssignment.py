# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 23:28:16 2020

@author: Media
"""

# Classification with Python
"""
In this notebook we try to practice all the classification algorithms that we learned in this course.

We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this 
specific dataset by accuracy evaluation methods.

Lets first load required libraries:
"""

# Preprocessing

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

# ABOUT DATASET
"""
This dataset is about past loans. The Loan_train.csv data set includes details of 346 customers whose loan 
are already paid off or defaulted. It includes following fields:
    
Field	                             Description
Loan_status	                         Whether a loan is paid off on in collection
Principal	                         Basic principal loan amount at the
Terms	                             Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule
Effective_date	                     When the loan got originated and took effects
Due_date	                         Since it’s one-time payoff schedule, each loan has one single due date
Age	                                 Age of applicant
Education	                         Education of applicant
Gender	                             The gender of applicant
"""

df = pd.read_csv('C:\\Users\\Media\\Desktop\\Coursera\\Machine Learning with Python - IBM\\Lab\\Week 6\\loan_train.csv')
df.head()

df.shape

"Convert to date time object"

df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()

# Data visualization and pre-processing
"Let’s see how many of each class is in our data set"

df['loan_status'].value_counts()

"""
260 people have paid off the loan on time while 86 have gone into collection

Lets plot some columns to underestand data better:
"""
# notice: install seaborn 
#!conda install -c anaconda seaborn -y
import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

"""FOR ME by ALESSANDRO
# Python program to demonstrate working 
# of map. 

# Return double of n 
def addition(n): 
    return n + n 
  
# We double all numbers using map() 
numbers = (1, 2, 3, 4) 
result = map(addition, numbers) 
print(list(result))
"""

# Pre-processing: Feature selection/extraction
"Lets look at the day of the week people get the loan"

df['dayofweek'] = df['effective_date'].dt.dayofweek      # Pandas.series.dt.dayofweek = returns day of the week as Monday 0 and Sunday 6

bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

"""
We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization 
to set a threshold values less then day 4
"""

df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


""" FOR ME by ALE
A lambda function is a small anonymous function.

A lambda function can take any number of arguments, but can only have one expression.

Syntax
lambda arguments : expression
The expression is executed and the result is returned:

Example
A lambda function that adds 10 to the number passed in as an argument, and print the result:

x = lambda a : a + 10
print(x(5))
Lambda functions can take any number of arguments:

Example
A lambda function that multiplies argument a with argument b and print the result:

x = lambda a, b : a * b
print(x(5, 6))

"""

# Convert Categorical features to numerical values
"Lets look at gender:"

df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)    # pandas.DataFrame.groupby --> Group DataFrame using a mapper or by a Series of columns.

"""
86 % of female pay their loan while only 73 % of males pay their loan

Lets convert male to 0 and female to 1:
"""

df['Gender'].replace(to_replace=['male','female'], value=[0,1], inplace=True)
df.head()

# One Hot Encoding
"How about education?"

df.groupby(['education'])['loan_status'].value_counts(normalize=True)

"Feature before One Hot Encoding"

df[['Principal','terms','age','Gender','education']].head()

"""
Use one hot encoding technique to convert categorical varables to binary variables and append them to the 
feature Data Frame
"""

Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

# Feature selection
"Lets define feature sets, X:"

X = Feature
X[0:5]

"What are our lables?"

y = df['loan_status'].values
y[0:5]

# Normalize Data
"Data Standardization give data zero mean and unit variance (technically should be done after train test split)"

X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

# Classification
"""
Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the 
accuracy of the model You should use the following algorithm:

- K Nearest Neighbor(KNN)
- Decision Tree
- Support Vector Machine
- Logistic Regression

__ Notice:__

- You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a 
  better model.
- You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
- You should include the code of the algorithm in the following cells.
"""

## K Nearest Neighbor(KNN)
"""
Notice: You should find the best k to build the model with the best accuracy.
warning: You should not use the loan_test.csv for finding the best k, however, you can split your 
train_loan.csv into train and test to find the best k.
"""
#######################################################################################
"FROM NOW MY EXAM"
#######################################################################################

# Train Test Split

"""
Out of Sample Accuracy is the percentage of correct predictions that the model makes on data that that the model has NOT been 
trained on. Doing a train and test on the same dataset will most likely have low out-of-sample accuracy, due to the likelihood 
of being over-fit.

It is important that our models have a high, out-of-sample accuracy, because the purpose of any model, of course, is to make 
correct predictions on unknown data. So how can we improve out-of-sample accuracy? One way is to use an evaluation approach 
called Train/Test Split. Train/Test Split involves splitting the dataset into training and testing sets respectively, which 
are mutually exclusive. After which, you train with the training set and test with the testing set.

This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset 
that have been used to train the data. It is more realistic for real world problems.
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Classification
"K nearest neighbor (KNN)"

# Import library
"Classifier implementing the k-nearest neighbors vote."

from sklearn.neighbors import KNeighborsClassifier

# Training
"Lets start the algorithm with k=4 for now:"

k = 4

#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

# Predicting
"We can use the model to predict the test set:"

yhat = neigh.predict(X_test)
yhat[0:5]

"""
In multilabel classification, accuracy classification score is a function that computes subset accuracy. This function is equal 
to the jaccard_similarity_score function. Essentially, it calculates how closely the actual labels and predicted labels are 
matched in the test set.
"""

from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

"We build the model again for multiple values of k from 1 to 10"

#ks = []
#TrainAcc = []
#TestAcc = []
#
#for k in range(1, 11):
#    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
#    yhat = neigh.predict(X_test)
#    TrainAcc.append(metrics.accuracy_score(y_train, neigh.predict(X_train)))
#    TestAcc.append(metrics.accuracy_score(y_test, yhat))
#    ks.append(k)
#    
#
#plt.figure()
#plt.plot(ks, TrainAcc, linewidth=1, color = 'blue', label = 'Train set accuracy')
#plt.plot(ks, TestAcc, linewidth=1, color = 'red', label = 'Test set accuracy')
#plt.xlabel('k ', size=16)
#plt.ylabel('Accuracy', size = 16)
#plt.legend(loc=1, prop={'size':12})

"We see the the accuracy of the test set is higher when k = 7 (getting closer to the accuracy of the training set)"

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1, Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

"Plot model accuracy for Different number of Neighbors:"

plt.figure()
plt.plot(range(1,Ks), mean_acc, 'blue')
plt.fill_between(range(1,Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors, K')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

#########################################################################################
## Decision Tree

from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Modeling
"""
We will first create an instance of the DecisionTreeClassifier called drugTree.
Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
"""

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters

"Next, we will fit the data with the training feature matrix X_trainset and training response vector y_trainset"

drugTree.fit(X_train,y_train)

# Prediction
"Let's make some predictions on the testing dataset and store it into a variable called predTree."

predTree = drugTree.predict(X_test)

"You can print out predTree and y_testset if you want to visually compare the prediction to the actual values."

print (predTree [0:5])
print (y_test [0:5])

# Evaluation
"Next, let's import metrics from sklearn and check the accuracy of our model."

from sklearn import metrics
import matplotlib.pyplot as plt

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))

# Visualization
"Lets visualize the tree"
# Notice: You might need to uncomment and install the pydotplus and graphviz libraries if you have not installed these before
# !conda install -c conda-forge pydotplus -y
# !conda install -c conda-forge python-graphviz -y

from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:8]
targetNames = df["loan_status"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


#########################################################################################
## Support Vector Machine

# Train/Test dataset
"Okay, we split our dataset into train and test set:"

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Modeling (SVM with Scikit-learn)
"""
The SVM algorithm offers a choice of kernel functions for performing its processing. Basically, mapping data 
into a higher dimensional space is called kernelling. The mathematical function used for the transformation 
is known as the kernel function, and can be of different types, such as:

1.Linear
2.Polynomial
3.Radial basis function (RBF)
4.Sigmoid

Each of these functions has its characteristics, its pros and cons, and its equation, but as there's no easy 
way of knowing which function performs best with any given dataset, we usually choose different functions in 
turn and compare the results. Let's just use the default, RBF (Radial Basis Function) for this lab.
"""

from sklearn import svm

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 

"After being fitted, the model can then be used to predict new values:"

yhat = clf.predict(X_test)
yhat [0:5]

#Evaluation

from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

"You can also easily use the f1_score from sklearn library:"

from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted')

"Lets try jaccard index for accuracy:"

from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)

# Practice
"""
Can you rebuild the model, but this time with a __linear__ kernel? You can use __kernel='linear'__ option, 
when you define the svm. How the accuracy changes with the new kernel function?
"""

clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train) 
yhat2 = clf2.predict(X_test)
print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard score: %.4f" % jaccard_similarity_score(y_test, yhat2))

