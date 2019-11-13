# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:18:08 2019

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:35:30 2019

@author: Stephany 
"""

## Data Preprocessing ##

## Importing the libraries ##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## Importing the dataset ##
#Loading the data set from the csv files
train_data = pd.read_csv('../titanic/train.csv')
test_data = pd.read_csv('../titanic/test.csv')


##Exploratory Data Analysis

print('train dataset: %s, test dataset %s' %(str(train_dataset.shape), str(test_dataset.shape)) )
train_dataset.head()


test_data = pd.read_csv('../titanic/test.csv')
test_data.head()


#Train Tune and Ensemble machine learning models


women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women ho survived:",rate_women)







































## Taking care of missing data ##
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan , strategy = 'mean' )
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:,1:3])

## Encoding categorical data ##
#Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform (X).toarray()

#Encoding the dependent variable
labelencoder_y=LabelEncoder()
y = labelencoder_y.fit_transform(y)

## Splitting the data set into training set and the test set ##
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2 ,random_state=0)


## Feature Scaling ##
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

##How to handel missing data


